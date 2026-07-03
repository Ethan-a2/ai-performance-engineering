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

static const char * BASELINE_KERNEL = R"CLC(
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

static double benchmark_ms(cl_command_queue queue, int warmup, int iterations, void (*enqueue)(void *), void * ctx) {
    for (int i = 0; i < warmup; ++i) {
        enqueue(ctx);
    }
    CL_CHECK(clFinish(queue));

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        enqueue(ctx);
    }
    CL_CHECK(clFinish(queue));
    const auto end = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / double(iterations);
}

struct BaselineRun {
    cl_command_queue queue;
    cl_kernel kernel;
    cl_mem weights;
    cl_mem src;
    cl_mem dst;
    int M;
    int N;
    int K;
};

static void enqueue_baseline(void * opaque) {
    BaselineRun * run = static_cast<BaselineRun *>(opaque);
    size_t local[2] = {16, 16};
    size_t global[2] = {round_up(size_t(run->M), local[0]), round_up(size_t(run->N), local[1])};
    CL_CHECK(clEnqueueNDRangeKernel(run->queue, run->kernel, 2, nullptr, global, local, 0, nullptr, nullptr));
}

struct XmemRun {
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

static void enqueue_xmem(void * opaque) {
    XmemRun * run = static_cast<XmemRun *>(opaque);
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

static float max_abs_error(const std::vector<float> & actual, const std::vector<float> & expected) {
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    }
    return max_error;
}

int main(int argc, char ** argv) {
    const char * xmem_kernel_path = argc > 1 ? argv[1] : "gemm_xmem_f16_f32_os8.cl";
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 30;
    const int M = argc > 3 ? std::atoi(argv[3]) : 1024;
    const int N = argc > 4 ? std::atoi(argv[4]) : 128;
    const int K = argc > 5 ? std::atoi(argv[5]) : 1024;
    const int warmup = 3;
    const int os = 8;

    if (M < 64 || N < 16 || K < 64 || (K % 8) != 0) {
        std::fprintf(stderr, "shape must satisfy M>=64, N>=16, K>=64, K%%8==0\n");
        return 1;
    }
    const int kpack = K / 4;
    const int npack = int(ceil_div(size_t(M), size_t(4)));

    std::vector<cl_half> weights_f16(size_t(M) * K);
    std::vector<float> src_f32(size_t(N) * K);
    std::vector<float> baseline_out(size_t(N) * M, 0.0f);
    std::vector<float> optimized_out(size_t(N) * M, 0.0f);
    std::vector<float> reference(size_t(N) * M, 0.0f);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float value = float((m * 7 + k * 3) % 17 - 8) * 0.005f;
            weights_f16[size_t(m) * K + k] = f32_to_f16(value);
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

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    select_adreno_device(platform, device);

    const std::string device_name = get_device_string(device, CL_DEVICE_NAME);
    const std::string extensions = get_device_string(device, CL_DEVICE_EXTENSIONS);
    std::printf("selected_device = %s\n", device_name.c_str());
    if (extensions.find("cl_qcom_subgroup_constant_load") == std::string::npos) {
        std::printf("qcom_extension_advertised = 0\n");
    } else {
        std::printf("qcom_extension_advertised = 1\n");
    }

    cl_int err = CL_SUCCESS;
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform),
        0,
    };
    cl_context context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    CL_CHECK(err);

    cl_program baseline_program = build_program(context, device, BASELINE_KERNEL);
    cl_program xmem_program = build_program(context, device, read_file(xmem_kernel_path));

    cl_kernel baseline_kernel = clCreateKernel(baseline_program, "baseline_gemm_f16_f32", &err);
    CL_CHECK(err);
    cl_kernel prepack_kernel = clCreateKernel(xmem_program, "adreno_xmem_prepack_weight_f16", &err);
    CL_CHECK(err);
    cl_kernel pack_src_kernel = clCreateKernel(xmem_program, "adreno_xmem_pack_src_f32", &err);
    CL_CHECK(err);
    cl_kernel xmem_kernel = clCreateKernel(xmem_program, "kernel_gemm_xmem_f16_f32_os8", &err);
    CL_CHECK(err);
    cl_kernel store_kernel = clCreateKernel(xmem_program, "adreno_xmem_store_dst_f32", &err);
    CL_CHECK(err);

    cl_mem weights_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        weights_f16.size() * sizeof(cl_half), weights_f16.data(), &err);
    CL_CHECK(err);
    cl_mem src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    src_f32.size() * sizeof(float), src_f32.data(), &err);
    CL_CHECK(err);
    cl_mem baseline_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         baseline_out.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem optimized_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                          optimized_out.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    const size_t packed_weight_bytes = size_t(kpack) * npack * 16 * sizeof(cl_half);
    cl_mem packed_weights = clCreateBuffer(context, CL_MEM_READ_WRITE, packed_weight_bytes, nullptr, &err);
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

    BaselineRun baseline_run = {queue, baseline_kernel, weights_buf, src_buf, baseline_dst, M, N, K};
    CL_CHECK(clSetKernelArg(baseline_kernel, 0, sizeof(cl_mem), &weights_buf));
    CL_CHECK(clSetKernelArg(baseline_kernel, 1, sizeof(cl_mem), &src_buf));
    CL_CHECK(clSetKernelArg(baseline_kernel, 2, sizeof(cl_mem), &baseline_dst));
    CL_CHECK(clSetKernelArg(baseline_kernel, 3, sizeof(int), &M));
    CL_CHECK(clSetKernelArg(baseline_kernel, 4, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(baseline_kernel, 5, sizeof(int), &K));

    XmemRun xmem_run = {
        queue, prepack_kernel, pack_src_kernel, xmem_kernel, store_kernel,
        weights_buf, src_buf, optimized_dst, packed_weights, xmem_buffer, src_img, dst_img,
        M, N, K, os, kpack, npack,
    };

    const double baseline_ms = benchmark_ms(queue, warmup, iterations, enqueue_baseline, &baseline_run);
    CL_CHECK(clEnqueueReadBuffer(queue, baseline_dst, CL_TRUE, 0,
                                 baseline_out.size() * sizeof(float), baseline_out.data(), 0, nullptr, nullptr));

    const double optimized_ms = benchmark_ms(queue, warmup, iterations, enqueue_xmem, &xmem_run);
    CL_CHECK(clEnqueueReadBuffer(queue, optimized_dst, CL_TRUE, 0,
                                 optimized_out.size() * sizeof(float), optimized_out.data(), 0, nullptr, nullptr));

    const float baseline_err = max_abs_error(baseline_out, reference);
    const float optimized_err = max_abs_error(optimized_out, reference);
    const double speedup = baseline_ms / optimized_ms;

    std::printf("=== Baseline: naive OpenCL GEMM ===\n");
    std::printf("baseline_ms = %.6f\n", baseline_ms);
    std::printf("baseline_max_abs_err = %.8f\n", baseline_err);
    std::printf("=== Optimized: llama.cpp Adreno xmem GEMM ===\n");
    std::printf("optimized_ms = %.6f\n", optimized_ms);
    std::printf("optimized_max_abs_err = %.8f\n", optimized_err);
    std::printf("speedup = %.6f\n", speedup);
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
    clReleaseProgram(xmem_program);
    clReleaseProgram(baseline_program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return optimized_err < 0.02f ? 0 : 2;
}
