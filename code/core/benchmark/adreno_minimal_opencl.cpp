#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const char * SIMPLE_KERNELS = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void baseline_gemm_f16_f32(
    __global const half * weights,
    __global const float * src,
    __global float * dst,
    int M,
    int N,
    int K) {
    const int m = get_global_id(0);
    const int n = get_global_id(1);
    if (m >= M || n >= N) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += (float)weights[m*K + k] * src[n*K + k];
    }
    dst[n*M + m] = sum;
}

__kernel void scale_kernel(__global const float * x, __global float * tmp, float scale, int count) {
    const int i = get_global_id(0);
    if (i < count) {
        tmp[i] = x[i] * scale;
    }
}

__kernel void bias_relu_kernel(__global const float * tmp, __global const float * bias, __global float * y, int cols, int count) {
    const int i = get_global_id(0);
    if (i < count) {
        const float value = tmp[i] + bias[i % cols];
        y[i] = fmax(value, 0.0f);
    }
}

__kernel void fused_scale_bias_relu_kernel(__global const float * x, __global const float * bias, __global float * y, float scale, int cols, int count) {
    const int i = get_global_id(0);
    if (i < count) {
        const float value = x[i] * scale + bias[i % cols];
        y[i] = fmax(value, 0.0f);
    }
}

__kernel void copy_scalar_kernel(__global const float * x, __global float * y, int count) {
    const int i = get_global_id(0);
    if (i < count) {
        y[i] = x[i];
    }
}

__kernel void copy_vector4_kernel(__global const float4 * x, __global float4 * y, int count4) {
    const int i = get_global_id(0);
    if (i < count4) {
        y[i] = x[i];
    }
}

__kernel void kv_token_update_kernel(__global const float * src, __global float * cache, int token, int stride) {
    const int i = get_global_id(0);
    if (i < stride) {
        cache[token*stride + i] = src[token*stride + i];
    }
}

__kernel void kv_block_update_kernel(__global const float * src, __global float * cache, int count) {
    const int i = get_global_id(0);
    if (i < count) {
        cache[i] = src[i];
    }
}
)CLC";

static size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static size_t round_up(size_t a, size_t b) {
    return ceil_div(a, b) * b;
}

static void check_cl(cl_int err, const char * expr) {
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "OpenCL error %d: %s\n", err, expr);
        std::exit(1);
    }
}

#define CL_CHECK(expr) check_cl((expr), #expr)

static std::string read_file(const char * path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "failed to open kernel file: %s\n", path);
        std::exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(file), {});
}

static std::string get_platform_string(cl_platform_id platform, cl_platform_info info) {
    size_t size = 0;
    CL_CHECK(clGetPlatformInfo(platform, info, 0, nullptr, &size));
    std::string out(size, '\0');
    CL_CHECK(clGetPlatformInfo(platform, info, size, out.data(), nullptr));
    return out;
}

static std::string get_device_string(cl_device_id device, cl_device_info info) {
    size_t size = 0;
    CL_CHECK(clGetDeviceInfo(device, info, 0, nullptr, &size));
    std::string out(size, '\0');
    CL_CHECK(clGetDeviceInfo(device, info, size, out.data(), nullptr));
    return out;
}

static cl_half f32_to_f16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = int((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return cl_half(sign);
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return cl_half(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return cl_half(sign | 0x7c00u);
    }
    return cl_half(sign | (uint32_t(exp) << 10) | ((mant + 0x1000u) >> 13));
}

static float f16_to_f32(cl_half value) {
    uint32_t sign = uint32_t(value & 0x8000u) << 16;
    uint32_t exp = (value >> 10) & 0x1fu;
    uint32_t mant = value & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static cl_program build_program(cl_context ctx, cl_device_id dev, const std::string & src) {
    const char * ptr = src.c_str();
    size_t len = src.size();
    cl_int err = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(ctx, 1, &ptr, &len, &err);
    CL_CHECK(err);
    const char * opts = "-cl-std=CL2.0 -cl-mad-enable -cl-unsafe-math-optimizations "
                        "-cl-finite-math-only -cl-fast-relaxed-math";
    err = clBuildProgram(program, 1, &dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
        std::fprintf(stderr, "build failed:\n%s\n", log.c_str());
        std::exit(1);
    }
    return program;
}

static bool contains_adreno_name(const std::string & value) {
    return value.find("QUALCOMM") != std::string::npos ||
           value.find("Qualcomm") != std::string::npos ||
           value.find("Adreno") != std::string::npos;
}

static void select_adreno_device(cl_platform_id & platform_out, cl_device_id & device_out) {
    cl_uint n_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &n_platforms));
    std::vector<cl_platform_id> platforms(n_platforms);
    CL_CHECK(clGetPlatformIDs(n_platforms, platforms.data(), nullptr));
    for (cl_platform_id platform : platforms) {
        cl_uint n_devices = 0;
        cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devices);
        if (err != CL_SUCCESS || n_devices == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(n_devices);
        CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, n_devices, devices.data(), nullptr));
        for (cl_device_id device : devices) {
            const std::string platform_name = get_platform_string(platform, CL_PLATFORM_NAME);
            const std::string platform_vendor = get_platform_string(platform, CL_PLATFORM_VENDOR);
            const std::string device_name = get_device_string(device, CL_DEVICE_NAME);
            if (contains_adreno_name(platform_name + platform_vendor + device_name)) {
                platform_out = platform;
                device_out = device;
                return;
            }
        }
    }
    std::fprintf(stderr, "no Adreno/QUALCOMM OpenCL GPU device found\n");
    std::exit(1);
}

static double time_ms(cl_command_queue queue, int warmup, int iterations, void (*enqueue)(void *), void * data) {
    for (int i = 0; i < warmup; ++i) {
        enqueue(data);
    }
    CL_CHECK(clFinish(queue));
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        enqueue(data);
    }
    CL_CHECK(clFinish(queue));
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / double(iterations);
}

static float max_abs_error(const std::vector<float> & a, const std::vector<float> & b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::fabs(a[i] - b[i]));
    }
    return err;
}

struct BaselineGemmRun {
    cl_command_queue queue;
    cl_kernel kernel;
    int M;
    int N;
};

static void enqueue_baseline_gemm(void * opaque) {
    BaselineGemmRun * run = static_cast<BaselineGemmRun *>(opaque);
    size_t local[2] = {16, 16};
    size_t global[2] = {round_up(size_t(run->M), local[0]), round_up(size_t(run->N), local[1])};
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->kernel, 2, nullptr, global, local, 0, nullptr, nullptr));
}

struct XmemGemmRun {
    cl_command_queue queue;
    cl_kernel prepack;
    cl_kernel pack_src;
    cl_kernel gemm;
    cl_kernel store;
    cl_mem weights;
    cl_mem src;
    cl_mem dst;
    cl_mem packed_weights;
    cl_mem xmem;
    cl_mem src_img;
    cl_mem dst_img;
    int M;
    int N;
    int K;
    int os;
    int kpack;
    int npack;
};

static void enqueue_xmem_gemm(void * opaque) {
    XmemGemmRun * run = static_cast<XmemGemmRun *>(opaque);
    cl_ulong zero_offset = 0;
    CL_CHECK(clSetKernelArg(run->prepack, 0, sizeof(cl_mem), &run->packed_weights));
    CL_CHECK(clSetKernelArg(run->prepack, 1, sizeof(cl_mem), &run->weights));
    CL_CHECK(clSetKernelArg(run->prepack, 2, sizeof(cl_ulong), &zero_offset));
    CL_CHECK(clSetKernelArg(run->prepack, 3, sizeof(int), &run->K));
    CL_CHECK(clSetKernelArg(run->prepack, 4, sizeof(int), &run->M));
    CL_CHECK(clSetKernelArg(run->prepack, 5, sizeof(int), &run->kpack));
    CL_CHECK(clSetKernelArg(run->prepack, 6, sizeof(int), &run->npack));
    CL_CHECK(clSetKernelArg(run->prepack, 7, sizeof(int), &run->os));
    size_t prepack_lws = 256;
    size_t prepack_gws = round_up(size_t(run->kpack) * run->npack, prepack_lws);
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->prepack, 1, nullptr, &prepack_gws, &prepack_lws, 0, nullptr, nullptr));

    CL_CHECK(clSetKernelArg(run->pack_src, 0, sizeof(cl_mem), &run->src));
    CL_CHECK(clSetKernelArg(run->pack_src, 1, sizeof(cl_ulong), &zero_offset));
    CL_CHECK(clSetKernelArg(run->pack_src, 2, sizeof(cl_mem), &run->src_img));
    CL_CHECK(clSetKernelArg(run->pack_src, 3, sizeof(int), &run->K));
    CL_CHECK(clSetKernelArg(run->pack_src, 4, sizeof(int), &run->N));
    size_t pack_lws[2] = {16, 16};
    size_t pack_gws[2] = {round_up(size_t(run->N), pack_lws[0]), round_up(size_t(run->kpack), pack_lws[1])};
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->pack_src, 2, nullptr, pack_gws, pack_lws, 0, nullptr, nullptr));

    CL_CHECK(clSetKernelArg(run->gemm, 0, sizeof(cl_mem), &run->packed_weights));
    CL_CHECK(clSetKernelArg(run->gemm, 1, sizeof(cl_mem), &run->xmem));
    CL_CHECK(clSetKernelArg(run->gemm, 2, sizeof(cl_mem), &run->src_img));
    CL_CHECK(clSetKernelArg(run->gemm, 3, sizeof(cl_mem), &run->dst_img));
    CL_CHECK(clSetKernelArg(run->gemm, 4, sizeof(int), &run->N));
    CL_CHECK(clSetKernelArg(run->gemm, 5, sizeof(int), &run->npack));
    CL_CHECK(clSetKernelArg(run->gemm, 6, sizeof(int), &run->kpack));
    const size_t z_values = ceil_div(size_t(run->npack), size_t(run->os));
    size_t gemm_lws[3] = {64, 1, 1};
    size_t gemm_gws[3] = {z_values * gemm_lws[0], ceil_div(size_t(run->N), gemm_lws[0]), 1};
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->gemm, 3, nullptr, gemm_gws, gemm_lws, 0, nullptr, nullptr));

    CL_CHECK(clSetKernelArg(run->store, 0, sizeof(cl_mem), &run->dst_img));
    CL_CHECK(clSetKernelArg(run->store, 1, sizeof(cl_mem), &run->dst));
    CL_CHECK(clSetKernelArg(run->store, 2, sizeof(cl_ulong), &zero_offset));
    CL_CHECK(clSetKernelArg(run->store, 3, sizeof(int), &run->M));
    CL_CHECK(clSetKernelArg(run->store, 4, sizeof(int), &run->N));
    size_t store_lws[2] = {16, 16};
    size_t store_gws[2] = {round_up(size_t(run->N), store_lws[0]), round_up(size_t(run->npack), store_lws[1])};
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->store, 2, nullptr, store_gws, store_lws, 0, nullptr, nullptr));
}

static int run_xmem_gemm(cl_context context, cl_command_queue queue, cl_program simple_program, cl_program xmem_program, int iterations) {
    const int M = 1024;
    const int N = 128;
    const int K = 1024;
    const int os = 8;
    const int kpack = K / 4;
    const int npack = int(ceil_div(size_t(M), size_t(4)));

    std::vector<cl_half> weights_f16(size_t(M) * K);
    std::vector<float> src_f32(size_t(N) * K);
    std::vector<float> baseline_out(size_t(N) * M, 0.0f);
    std::vector<float> optimized_out(size_t(N) * M, 0.0f);
    std::vector<float> reference(size_t(N) * M, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            weights_f16[size_t(m) * K + k] = f32_to_f16(float((m * 7 + k * 3) % 17 - 8) * 0.005f);
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            src_f32[size_t(n) * K + k] = float((n * 5 + k * 11) % 19 - 9) * 0.005f;
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += f16_to_f32(weights_f16[size_t(m) * K + k]) * src_f32[size_t(n) * K + k];
            }
            reference[size_t(n) * M + m] = sum;
        }
    }

    cl_int err = CL_SUCCESS;
    cl_kernel baseline_kernel = clCreateKernel(simple_program, "baseline_gemm_f16_f32", &err);
    CL_CHECK(err);
    cl_kernel prepack_kernel = clCreateKernel(xmem_program, "adreno_xmem_prepack_weight_f16", &err);
    CL_CHECK(err);
    cl_kernel pack_src_kernel = clCreateKernel(xmem_program, "adreno_xmem_pack_src_f32", &err);
    CL_CHECK(err);
    cl_kernel xmem_kernel = clCreateKernel(xmem_program, "kernel_gemm_xmem_f16_f32_os8", &err);
    CL_CHECK(err);
    cl_kernel store_kernel = clCreateKernel(xmem_program, "adreno_xmem_store_dst_f32", &err);
    CL_CHECK(err);

    cl_mem weights_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_f16.size() * sizeof(cl_half), weights_f16.data(), &err);
    CL_CHECK(err);
    cl_mem src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src_f32.size() * sizeof(float), src_f32.data(), &err);
    CL_CHECK(err);
    cl_mem baseline_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, baseline_out.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem optimized_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, optimized_out.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem packed_weights = clCreateBuffer(context, CL_MEM_READ_WRITE, size_t(kpack) * npack * 16 * sizeof(cl_half), nullptr, &err);
    CL_CHECK(err);
    cl_mem xmem_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 6144, nullptr, &err);
    CL_CHECK(err);

    cl_image_format image_format = {};
    image_format.image_channel_order = CL_RGBA;
    image_format.image_channel_data_type = CL_HALF_FLOAT;
    cl_image_desc src_desc = {};
    src_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    src_desc.image_width = N;
    src_desc.image_height = kpack;
    cl_mem src_img = clCreateImage(context, CL_MEM_READ_WRITE, &image_format, &src_desc, nullptr, &err);
    CL_CHECK(err);
    cl_image_desc dst_desc = {};
    dst_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dst_desc.image_width = N;
    dst_desc.image_height = npack;
    cl_mem dst_img = clCreateImage(context, CL_MEM_READ_WRITE, &image_format, &dst_desc, nullptr, &err);
    CL_CHECK(err);

    CL_CHECK(clSetKernelArg(baseline_kernel, 0, sizeof(cl_mem), &weights_buf));
    CL_CHECK(clSetKernelArg(baseline_kernel, 1, sizeof(cl_mem), &src_buf));
    CL_CHECK(clSetKernelArg(baseline_kernel, 2, sizeof(cl_mem), &baseline_dst));
    CL_CHECK(clSetKernelArg(baseline_kernel, 3, sizeof(int), &M));
    CL_CHECK(clSetKernelArg(baseline_kernel, 4, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(baseline_kernel, 5, sizeof(int), &K));
    BaselineGemmRun baseline_run = {queue, baseline_kernel, M, N};
    XmemGemmRun xmem_run = {queue, prepack_kernel, pack_src_kernel, xmem_kernel, store_kernel, weights_buf, src_buf, optimized_dst, packed_weights, xmem_buffer, src_img, dst_img, M, N, K, os, kpack, npack};

    const double baseline_ms = time_ms(queue, 3, iterations, enqueue_baseline_gemm, &baseline_run);
    CL_CHECK(clEnqueueReadBuffer(queue, baseline_dst, CL_TRUE, 0, baseline_out.size() * sizeof(float), baseline_out.data(), 0, nullptr, nullptr));
    const double optimized_ms = time_ms(queue, 3, iterations, enqueue_xmem_gemm, &xmem_run);
    CL_CHECK(clEnqueueReadBuffer(queue, optimized_dst, CL_TRUE, 0, optimized_out.size() * sizeof(float), optimized_out.data(), 0, nullptr, nullptr));
    const float baseline_err = max_abs_error(baseline_out, reference);
    const float optimized_err = max_abs_error(optimized_out, reference);
    std::printf("baseline_ms = %.6f\n", baseline_ms);
    std::printf("optimized_ms = %.6f\n", optimized_ms);
    std::printf("speedup = %.6f\n", baseline_ms / optimized_ms);
    std::printf("baseline_max_abs_err = %.8f\n", baseline_err);
    std::printf("optimized_max_abs_err = %.8f\n", optimized_err);
    std::printf("shape = M:%d N:%d K:%d iterations:%d\n", M, N, K, iterations);

    clReleaseMemObject(dst_img);
    clReleaseMemObject(src_img);
    clReleaseMemObject(xmem_buffer);
    clReleaseMemObject(packed_weights);
    clReleaseMemObject(optimized_dst);
    clReleaseMemObject(baseline_dst);
    clReleaseMemObject(src_buf);
    clReleaseMemObject(weights_buf);
    clReleaseKernel(store_kernel);
    clReleaseKernel(xmem_kernel);
    clReleaseKernel(pack_src_kernel);
    clReleaseKernel(prepack_kernel);
    clReleaseKernel(baseline_kernel);
    return optimized_err < 0.001f ? 0 : 2;
}

struct FusionRun {
    cl_command_queue queue;
    cl_kernel first;
    cl_kernel second;
    cl_kernel fused;
    size_t global;
    size_t local;
};

static void enqueue_fusion_baseline(void * opaque) {
    FusionRun * run = static_cast<FusionRun *>(opaque);
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->first, 1, nullptr, &run->global, &run->local, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->second, 1, nullptr, &run->global, &run->local, 0, nullptr, nullptr));
}

static void enqueue_fusion_optimized(void * opaque) {
    FusionRun * run = static_cast<FusionRun *>(opaque);
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->fused, 1, nullptr, &run->global, &run->local, 0, nullptr, nullptr));
}

static int run_fusion(cl_context context, cl_command_queue queue, cl_program program, int iterations) {
    const int rows = 4096;
    const int cols = 256;
    const int count = rows * cols;
    const float scale = 1.125f;
    std::vector<float> x(count);
    std::vector<float> bias(cols);
    std::vector<float> baseline(count, 0.0f);
    std::vector<float> optimized(count, 0.0f);
    for (int i = 0; i < count; ++i) {
        x[i] = float((i * 13) % 127 - 63) * 0.001f;
    }
    for (int i = 0; i < cols; ++i) {
        bias[i] = float((i * 17) % 29 - 14) * 0.001f;
    }
    cl_int err = CL_SUCCESS;
    cl_kernel scale_kernel = clCreateKernel(program, "scale_kernel", &err);
    CL_CHECK(err);
    cl_kernel bias_relu_kernel = clCreateKernel(program, "bias_relu_kernel", &err);
    CL_CHECK(err);
    cl_kernel fused_kernel = clCreateKernel(program, "fused_scale_bias_relu_kernel", &err);
    CL_CHECK(err);
    cl_mem x_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.size() * sizeof(float), x.data(), &err);
    CL_CHECK(err);
    cl_mem bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bias.size() * sizeof(float), bias.data(), &err);
    CL_CHECK(err);
    cl_mem tmp_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, x.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem baseline_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, x.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem optimized_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, x.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(scale_kernel, 0, sizeof(cl_mem), &x_buf));
    CL_CHECK(clSetKernelArg(scale_kernel, 1, sizeof(cl_mem), &tmp_buf));
    CL_CHECK(clSetKernelArg(scale_kernel, 2, sizeof(float), &scale));
    CL_CHECK(clSetKernelArg(scale_kernel, 3, sizeof(int), &count));
    CL_CHECK(clSetKernelArg(bias_relu_kernel, 0, sizeof(cl_mem), &tmp_buf));
    CL_CHECK(clSetKernelArg(bias_relu_kernel, 1, sizeof(cl_mem), &bias_buf));
    CL_CHECK(clSetKernelArg(bias_relu_kernel, 2, sizeof(cl_mem), &baseline_buf));
    CL_CHECK(clSetKernelArg(bias_relu_kernel, 3, sizeof(int), &cols));
    CL_CHECK(clSetKernelArg(bias_relu_kernel, 4, sizeof(int), &count));
    CL_CHECK(clSetKernelArg(fused_kernel, 0, sizeof(cl_mem), &x_buf));
    CL_CHECK(clSetKernelArg(fused_kernel, 1, sizeof(cl_mem), &bias_buf));
    CL_CHECK(clSetKernelArg(fused_kernel, 2, sizeof(cl_mem), &optimized_buf));
    CL_CHECK(clSetKernelArg(fused_kernel, 3, sizeof(float), &scale));
    CL_CHECK(clSetKernelArg(fused_kernel, 4, sizeof(int), &cols));
    CL_CHECK(clSetKernelArg(fused_kernel, 5, sizeof(int), &count));
    size_t local = 256;
    size_t global = round_up(size_t(count), local);
    FusionRun run = {queue, scale_kernel, bias_relu_kernel, fused_kernel, global, local};
    const double baseline_ms = time_ms(queue, 5, iterations, enqueue_fusion_baseline, &run);
    CL_CHECK(clEnqueueReadBuffer(queue, baseline_buf, CL_TRUE, 0, baseline.size() * sizeof(float), baseline.data(), 0, nullptr, nullptr));
    const double optimized_ms = time_ms(queue, 5, iterations, enqueue_fusion_optimized, &run);
    CL_CHECK(clEnqueueReadBuffer(queue, optimized_buf, CL_TRUE, 0, optimized.size() * sizeof(float), optimized.data(), 0, nullptr, nullptr));
    const float error = max_abs_error(baseline, optimized);
    std::printf("baseline_ms = %.6f\n", baseline_ms);
    std::printf("optimized_ms = %.6f\n", optimized_ms);
    std::printf("speedup = %.6f\n", baseline_ms / optimized_ms);
    std::printf("optimized_max_abs_err = %.8f\n", error);
    std::printf("shape = rows:%d cols:%d iterations:%d\n", rows, cols, iterations);
    clReleaseMemObject(optimized_buf);
    clReleaseMemObject(baseline_buf);
    clReleaseMemObject(tmp_buf);
    clReleaseMemObject(bias_buf);
    clReleaseMemObject(x_buf);
    clReleaseKernel(fused_kernel);
    clReleaseKernel(bias_relu_kernel);
    clReleaseKernel(scale_kernel);
    return error < 0.0001f ? 0 : 2;
}

struct CopyRun {
    cl_command_queue queue;
    cl_kernel scalar_first;
    cl_kernel scalar_second;
    cl_kernel vector4;
    size_t scalar_global;
    size_t vector_global;
    size_t local;
};

static void enqueue_copy_baseline(void * opaque) {
    CopyRun * run = static_cast<CopyRun *>(opaque);
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->scalar_first, 1, nullptr, &run->scalar_global, &run->local, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->scalar_second, 1, nullptr, &run->scalar_global, &run->local, 0, nullptr, nullptr));
}

static void enqueue_copy_optimized(void * opaque) {
    CopyRun * run = static_cast<CopyRun *>(opaque);
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->vector4, 1, nullptr, &run->vector_global, &run->local, 0, nullptr, nullptr));
}

static int run_copy_vectorized(cl_context context, cl_command_queue queue, cl_program program, int iterations) {
    const int count = 4 * 1024 * 1024;
    const int count4 = count / 4;
    std::vector<float> input(count);
    std::vector<float> baseline(count, 0.0f);
    std::vector<float> optimized(count, 0.0f);
    for (int i = 0; i < count; ++i) {
        input[i] = float((i * 19) % 257) * 0.001f;
    }
    cl_int err = CL_SUCCESS;
    cl_kernel scalar_first_kernel = clCreateKernel(program, "copy_scalar_kernel", &err);
    CL_CHECK(err);
    cl_kernel scalar_second_kernel = clCreateKernel(program, "copy_scalar_kernel", &err);
    CL_CHECK(err);
    cl_kernel vector_kernel = clCreateKernel(program, "copy_vector4_kernel", &err);
    CL_CHECK(err);
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), input.data(), &err);
    CL_CHECK(err);
    cl_mem baseline_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem tmp_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, input.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem optimized_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(scalar_first_kernel, 0, sizeof(cl_mem), &input_buf));
    CL_CHECK(clSetKernelArg(scalar_first_kernel, 1, sizeof(cl_mem), &tmp_buf));
    CL_CHECK(clSetKernelArg(scalar_first_kernel, 2, sizeof(int), &count));
    CL_CHECK(clSetKernelArg(scalar_second_kernel, 0, sizeof(cl_mem), &tmp_buf));
    CL_CHECK(clSetKernelArg(scalar_second_kernel, 1, sizeof(cl_mem), &baseline_buf));
    CL_CHECK(clSetKernelArg(scalar_second_kernel, 2, sizeof(int), &count));
    CL_CHECK(clSetKernelArg(vector_kernel, 0, sizeof(cl_mem), &input_buf));
    CL_CHECK(clSetKernelArg(vector_kernel, 1, sizeof(cl_mem), &optimized_buf));
    CL_CHECK(clSetKernelArg(vector_kernel, 2, sizeof(int), &count4));
    size_t local = 256;
    CopyRun run = {queue, scalar_first_kernel, scalar_second_kernel, vector_kernel, round_up(size_t(count), local), round_up(size_t(count4), local), local};
    const double baseline_ms = time_ms(queue, 5, iterations, enqueue_copy_baseline, &run);
    CL_CHECK(clEnqueueReadBuffer(queue, baseline_buf, CL_TRUE, 0, baseline.size() * sizeof(float), baseline.data(), 0, nullptr, nullptr));
    const double optimized_ms = time_ms(queue, 5, iterations, enqueue_copy_optimized, &run);
    CL_CHECK(clEnqueueReadBuffer(queue, optimized_buf, CL_TRUE, 0, optimized.size() * sizeof(float), optimized.data(), 0, nullptr, nullptr));
    const float error = max_abs_error(baseline, optimized);
    std::printf("baseline_ms = %.6f\n", baseline_ms);
    std::printf("optimized_ms = %.6f\n", optimized_ms);
    std::printf("speedup = %.6f\n", baseline_ms / optimized_ms);
    std::printf("optimized_max_abs_err = %.8f\n", error);
    std::printf("shape = elements:%d iterations:%d\n", count, iterations);
    clReleaseMemObject(optimized_buf);
    clReleaseMemObject(tmp_buf);
    clReleaseMemObject(baseline_buf);
    clReleaseMemObject(input_buf);
    clReleaseKernel(vector_kernel);
    clReleaseKernel(scalar_second_kernel);
    clReleaseKernel(scalar_first_kernel);
    return error < 0.0001f ? 0 : 2;
}

struct KvRun {
    cl_command_queue queue;
    cl_kernel token;
    cl_kernel block;
    int tokens;
    int stride;
    size_t token_global;
    size_t block_global;
    size_t local;
};

static void enqueue_kv_baseline(void * opaque) {
    KvRun * run = static_cast<KvRun *>(opaque);
    for (int token = 0; token < run->tokens; ++token) {
        CL_CHECK(clSetKernelArg(run->token, 2, sizeof(int), &token));
        CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->token, 1, nullptr, &run->token_global, &run->local, 0, nullptr, nullptr));
    }
}

static void enqueue_kv_optimized(void * opaque) {
    KvRun * run = static_cast<KvRun *>(opaque);
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->block, 1, nullptr, &run->block_global, &run->local, 0, nullptr, nullptr));
}

static int run_kv_block(cl_context context, cl_command_queue queue, cl_program program, int iterations) {
    const int tokens = 128;
    const int stride = 4096;
    const int count = tokens * stride;
    std::vector<float> input(count);
    std::vector<float> baseline(count, 0.0f);
    std::vector<float> optimized(count, 0.0f);
    for (int i = 0; i < count; ++i) {
        input[i] = float((i * 23) % 251) * 0.001f;
    }
    cl_int err = CL_SUCCESS;
    cl_kernel token_kernel = clCreateKernel(program, "kv_token_update_kernel", &err);
    CL_CHECK(err);
    cl_kernel block_kernel = clCreateKernel(program, "kv_block_update_kernel", &err);
    CL_CHECK(err);
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), input.data(), &err);
    CL_CHECK(err);
    cl_mem baseline_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem optimized_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    int zero_token = 0;
    CL_CHECK(clSetKernelArg(token_kernel, 0, sizeof(cl_mem), &input_buf));
    CL_CHECK(clSetKernelArg(token_kernel, 1, sizeof(cl_mem), &baseline_buf));
    CL_CHECK(clSetKernelArg(token_kernel, 2, sizeof(int), &zero_token));
    CL_CHECK(clSetKernelArg(token_kernel, 3, sizeof(int), &stride));
    CL_CHECK(clSetKernelArg(block_kernel, 0, sizeof(cl_mem), &input_buf));
    CL_CHECK(clSetKernelArg(block_kernel, 1, sizeof(cl_mem), &optimized_buf));
    CL_CHECK(clSetKernelArg(block_kernel, 2, sizeof(int), &count));
    size_t local = 256;
    KvRun run = {queue, token_kernel, block_kernel, tokens, stride, round_up(size_t(stride), local), round_up(size_t(count), local), local};
    const double baseline_ms = time_ms(queue, 3, iterations, enqueue_kv_baseline, &run);
    CL_CHECK(clEnqueueReadBuffer(queue, baseline_buf, CL_TRUE, 0, baseline.size() * sizeof(float), baseline.data(), 0, nullptr, nullptr));
    const double optimized_ms = time_ms(queue, 3, iterations, enqueue_kv_optimized, &run);
    CL_CHECK(clEnqueueReadBuffer(queue, optimized_buf, CL_TRUE, 0, optimized.size() * sizeof(float), optimized.data(), 0, nullptr, nullptr));
    const float error = max_abs_error(baseline, optimized);
    std::printf("baseline_ms = %.6f\n", baseline_ms);
    std::printf("optimized_ms = %.6f\n", optimized_ms);
    std::printf("speedup = %.6f\n", baseline_ms / optimized_ms);
    std::printf("optimized_max_abs_err = %.8f\n", error);
    std::printf("shape = tokens:%d stride:%d iterations:%d\n", tokens, stride, iterations);
    clReleaseMemObject(optimized_buf);
    clReleaseMemObject(baseline_buf);
    clReleaseMemObject(input_buf);
    clReleaseKernel(block_kernel);
    clReleaseKernel(token_kernel);
    return error < 0.0001f ? 0 : 2;
}

int main(int argc, char ** argv) {
    const std::string scenario = argc > 1 ? argv[1] : "xmem_gemm";
    const char * xmem_kernel_path = argc > 2 ? argv[2] : "gemm_xmem_f16_f32_os8.cl";
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 30;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    select_adreno_device(platform, device);
    const std::string device_name = get_device_string(device, CL_DEVICE_NAME);
    const std::string extensions = get_device_string(device, CL_DEVICE_EXTENSIONS);
    std::printf("scenario = %s\n", scenario.c_str());
    std::printf("selected_device = %s\n", device_name.c_str());
    std::printf("qcom_extension_advertised = %d\n", extensions.find("cl_qcom_subgroup_constant_load") == std::string::npos ? 0 : 1);

    cl_int err = CL_SUCCESS;
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};
    cl_context context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    CL_CHECK(err);
    cl_program simple_program = build_program(context, device, SIMPLE_KERNELS);
    int rc = 1;
    if (scenario == "xmem_gemm") {
        cl_program xmem_program = build_program(context, device, read_file(xmem_kernel_path));
        rc = run_xmem_gemm(context, queue, simple_program, xmem_program, iterations);
        clReleaseProgram(xmem_program);
    } else if (scenario == "pipeline_fusion") {
        rc = run_fusion(context, queue, simple_program, iterations);
    } else if (scenario == "copy_vectorized") {
        rc = run_copy_vectorized(context, queue, simple_program, iterations);
    } else if (scenario == "kv_block") {
        rc = run_kv_block(context, queue, simple_program, iterations);
    } else {
        std::fprintf(stderr, "unknown scenario: %s\n", scenario.c_str());
        rc = 1;
    }
    clReleaseProgram(simple_program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return rc;
}
