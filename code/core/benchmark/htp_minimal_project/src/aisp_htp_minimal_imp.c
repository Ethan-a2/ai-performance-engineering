#include "aisp_htp_minimal.h"

#include <AEEStdErr.h>
#include <HAP_power.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <remote.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

__attribute__((used, visibility("default"))) const char aisp_htp_minimal_ts[] =
    "\nTIMESTAMP=" __DATE__ " " __TIME__ "\n";

#define AISP_HTP_MINIMAL_MAGIC 0x4854504du
#define AISP_HTP_SCENARIO_HVX_TILE 0u
#define AISP_HTP_SCENARIO_PIPELINE_FUSION 1u
#define AISP_HTP_SCENARIO_COPY_VECTORIZED 2u
#define AISP_HTP_SCENARIO_KV_BLOCK 3u
#define AISP_HTP_MODE_BASELINE 0u
#define AISP_HTP_MODE_OPTIMIZED 1u

struct aisp_htp_minimal_context {
    uint32 magic;
};

static uint32 hvx_bytes(void) {
    return (uint32) sizeof(HVX_Vector);
}

static uint32 align_up_u32(uint32 value, uint32 alignment) {
    return ((value + alignment - 1u) / alignment) * alignment;
}

static void * aligned_malloc_128(size_t bytes) {
    const size_t extra = 127u + sizeof(void *);
    uint8_t * base = (uint8_t *) malloc(bytes + extra);
    if (!base) {
        return NULL;
    }
    uintptr_t aligned = ((uintptr_t) (base + sizeof(void *) + 127u)) & ~(uintptr_t) 127u;
    ((void **) aligned)[-1] = base;
    return (void *) aligned;
}

static void aligned_free_128(void * ptr) {
    if (ptr) {
        free(((void **) ptr)[-1]);
    }
}

static uint64 checksum_u32(const uint32_t * data, uint32 count) {
    uint64 acc = 1469598103934665603ULL;
    for (uint32 i = 0; i < count; ++i) {
        acc ^= (uint64) data[i] + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
    }
    return acc;
}

static uint64 checksum_u8(const uint8_t * data, uint32 count) {
    uint64 acc = 1469598103934665603ULL;
    for (uint32 i = 0; i < count; ++i) {
        acc ^= (uint64) data[i] + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
    }
    return acc;
}

static int request_power_votes(void * client) {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    int err = HAP_power_set(client, &request);
    if (err != AEE_SUCCESS) {
        return err;
    }

    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.set_dcvs_enable = 1;
    request.dcvs_v3.dcvs_enable = 0;
    request.dcvs_v3.set_bus_params = 1;
    request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_core_params = 1;
    request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_sleep_disable = 1;
    request.dcvs_v3.sleep_disable = 1;
    err = HAP_power_set(client, &request);
    if (err != AEE_SUCCESS) {
        return err;
    }

    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = 1;
    return HAP_power_set(client, &request);
}

int aisp_htp_minimal_open(const char * uri, remote_handle64 * handle) {
    (void) uri;
    struct aisp_htp_minimal_context * ctx = (struct aisp_htp_minimal_context *) calloc(1, sizeof(*ctx));
    if (!ctx) {
        return AEE_ENOMEMORY;
    }
    ctx->magic = AISP_HTP_MINIMAL_MAGIC;
    int err = request_power_votes(ctx);
    if (err != AEE_SUCCESS) {
        free(ctx);
        return err;
    }
    *handle = (remote_handle64) ctx;
    return AEE_SUCCESS;
}

int aisp_htp_minimal_close(remote_handle64 handle) {
    struct aisp_htp_minimal_context * ctx = (struct aisp_htp_minimal_context *) handle;
    if (!ctx || ctx->magic != AISP_HTP_MINIMAL_MAGIC) {
        return AEE_EBADHANDLE;
    }
    ctx->magic = 0;
    free(ctx);
    return AEE_SUCCESS;
}

int aisp_htp_minimal_hwinfo(remote_handle64 handle, uint32 * n_threads, uint32 * out_hvx_bytes, uint32 * arch) {
    struct aisp_htp_minimal_context * ctx = (struct aisp_htp_minimal_context *) handle;
    if (!ctx || ctx->magic != AISP_HTP_MINIMAL_MAGIC || !n_threads || !out_hvx_bytes || !arch) {
        return AEE_EBADPARM;
    }
    *n_threads = 1;
    *out_hvx_bytes = hvx_bytes();
#ifdef __HEXAGON_ARCH__
    *arch = (uint32) __HEXAGON_ARCH__;
#else
    *arch = 0;
#endif
    return AEE_SUCCESS;
}

static int run_hvx_tile(uint32 mode, uint32 elements, uint32 repeats, uint64 * checksum) {
    const uint32 lanes = hvx_bytes() / (uint32) sizeof(uint32_t);
    const uint32 n = align_up_u32(elements < lanes ? lanes : elements, lanes);
    uint32_t * a = (uint32_t *) aligned_malloc_128((size_t) n * sizeof(uint32_t));
    uint32_t * b = (uint32_t *) aligned_malloc_128((size_t) n * sizeof(uint32_t));
    uint32_t * out = (uint32_t *) aligned_malloc_128((size_t) n * sizeof(uint32_t));
    if (!a || !b || !out) {
        aligned_free_128(a);
        aligned_free_128(b);
        aligned_free_128(out);
        return AEE_ENOMEMORY;
    }

    for (uint32 i = 0; i < n; ++i) {
        a[i] = (uint32_t) ((i * 17u + 5u) & 1023u);
        b[i] = (uint32_t) ((i * 7u + 3u) & 1023u);
        out[i] = (uint32_t) (i & 15u);
    }

    if (mode == AISP_HTP_MODE_BASELINE) {
        for (uint32 r = 0; r < repeats; ++r) {
            const uint32_t bias = (uint32_t) ((r & 31u) + 1u);
            for (uint32 i = 0; i < n; ++i) {
                out[i] += a[i] + b[i] + bias;
            }
        }
    } else {
        HVX_Vector * av = (HVX_Vector *) a;
        HVX_Vector * bv = (HVX_Vector *) b;
        HVX_Vector * ov = (HVX_Vector *) out;
        const uint32 vectors = ((uint32) n * sizeof(uint32_t)) / hvx_bytes();
        for (uint32 r = 0; r < repeats; ++r) {
            const HVX_Vector bias = Q6_V_vsplat_R((int) ((r & 31u) + 1u));
            for (uint32 i = 0; i < vectors; ++i) {
                HVX_Vector sum = Q6_Vw_vadd_VwVw(av[i], bv[i]);
                sum = Q6_Vw_vadd_VwVw(sum, bias);
                ov[i] = Q6_Vw_vadd_VwVw(ov[i], sum);
            }
        }
    }

    *checksum = checksum_u32(out, n);
    aligned_free_128(a);
    aligned_free_128(b);
    aligned_free_128(out);
    return AEE_SUCCESS;
}

static int run_pipeline_fusion(uint32 mode, uint32 elements, uint32 repeats, uint64 * checksum) {
    const uint32 lanes = hvx_bytes() / (uint32) sizeof(uint32_t);
    const uint32 n = align_up_u32(elements < lanes ? lanes : elements, lanes);
    uint32_t * a = (uint32_t *) aligned_malloc_128((size_t) n * sizeof(uint32_t));
    uint32_t * b = (uint32_t *) aligned_malloc_128((size_t) n * sizeof(uint32_t));
    uint32_t * tmp = (uint32_t *) aligned_malloc_128((size_t) n * sizeof(uint32_t));
    uint32_t * out = (uint32_t *) aligned_malloc_128((size_t) n * sizeof(uint32_t));
    if (!a || !b || !tmp || !out) {
        aligned_free_128(a);
        aligned_free_128(b);
        aligned_free_128(tmp);
        aligned_free_128(out);
        return AEE_ENOMEMORY;
    }

    for (uint32 i = 0; i < n; ++i) {
        a[i] = (uint32_t) ((i * 13u + 9u) & 511u);
        b[i] = (uint32_t) ((i * 5u + 21u) & 511u);
        tmp[i] = 0;
        out[i] = (uint32_t) (i & 7u);
    }

    if (mode == AISP_HTP_MODE_BASELINE) {
        for (uint32 r = 0; r < repeats; ++r) {
            const uint32_t bias = (uint32_t) ((r & 7u) + 11u);
            for (uint32 i = 0; i < n; ++i) {
                tmp[i] = out[i] + a[i] + bias;
            }
            for (uint32 i = 0; i < n; ++i) {
                out[i] = tmp[i] + b[i] - bias;
            }
        }
    } else {
        HVX_Vector * av = (HVX_Vector *) a;
        HVX_Vector * bv = (HVX_Vector *) b;
        HVX_Vector * ov = (HVX_Vector *) out;
        const uint32 vectors = ((uint32) n * sizeof(uint32_t)) / hvx_bytes();
        for (uint32 r = 0; r < repeats; ++r) {
            (void) r;
            for (uint32 i = 0; i < vectors; ++i) {
                HVX_Vector sum = Q6_Vw_vadd_VwVw(av[i], bv[i]);
                ov[i] = Q6_Vw_vadd_VwVw(ov[i], sum);
            }
        }
    }

    *checksum = checksum_u32(out, n);
    aligned_free_128(a);
    aligned_free_128(b);
    aligned_free_128(tmp);
    aligned_free_128(out);
    return AEE_SUCCESS;
}

static int run_copy_vectorized(uint32 mode, uint32 elements, uint32 repeats, uint64 * checksum) {
    const uint32 bytes = align_up_u32(elements < hvx_bytes() ? hvx_bytes() : elements, hvx_bytes());
    uint8_t * src = (uint8_t *) aligned_malloc_128(bytes);
    uint8_t * dst = (uint8_t *) aligned_malloc_128(bytes);
    if (!src || !dst) {
        aligned_free_128(src);
        aligned_free_128(dst);
        return AEE_ENOMEMORY;
    }
    for (uint32 i = 0; i < bytes; ++i) {
        src[i] = (uint8_t) ((i * 29u + 7u) & 255u);
        dst[i] = 0;
    }

    uint64 guard = 0;
    if (mode == AISP_HTP_MODE_BASELINE) {
        for (uint32 r = 0; r < repeats; ++r) {
            for (uint32 i = 0; i < bytes; ++i) {
                dst[i] = src[i];
            }
            guard += dst[(r * 257u) % bytes];
            src[(r * 131u) % bytes] ^= (uint8_t) (r + 1u);
        }
    } else {
        HVX_Vector * sv = (HVX_Vector *) src;
        HVX_Vector * dv = (HVX_Vector *) dst;
        const uint32 vectors = bytes / hvx_bytes();
        for (uint32 r = 0; r < repeats; ++r) {
            for (uint32 i = 0; i < vectors; ++i) {
                dv[i] = sv[i];
            }
            guard += dst[(r * 257u) % bytes];
            src[(r * 131u) % bytes] ^= (uint8_t) (r + 1u);
        }
    }

    *checksum = checksum_u8(dst, bytes) ^ guard;
    aligned_free_128(src);
    aligned_free_128(dst);
    return AEE_SUCCESS;
}

static int run_kv_block(uint32 mode, uint32 elements, uint32 repeats, uint64 * checksum) {
    const uint32 lanes = hvx_bytes() / (uint32) sizeof(uint32_t);
    const uint32 width = align_up_u32(elements < lanes ? lanes : elements, lanes);
    const uint32 slots = 32u;
    const uint32 total = width * slots;
    uint32_t * update = (uint32_t *) aligned_malloc_128((size_t) width * sizeof(uint32_t));
    uint32_t * kv = (uint32_t *) aligned_malloc_128((size_t) total * sizeof(uint32_t));
    if (!update || !kv) {
        aligned_free_128(update);
        aligned_free_128(kv);
        return AEE_ENOMEMORY;
    }
    for (uint32 i = 0; i < width; ++i) {
        update[i] = (uint32_t) ((i * 3u + 17u) & 255u);
    }
    for (uint32 i = 0; i < total; ++i) {
        kv[i] = (uint32_t) (i & 31u);
    }

    if (mode == AISP_HTP_MODE_BASELINE) {
        for (uint32 r = 0; r < repeats; ++r) {
            for (uint32 slot = 0; slot < slots; ++slot) {
                const uint32_t delta = (uint32_t) (slot + (r & 31u));
                uint32_t * row = kv + (size_t) slot * width;
                for (uint32 i = 0; i < width; ++i) {
                    row[i] += update[i] + delta;
                }
            }
        }
    } else {
        HVX_Vector * uv = (HVX_Vector *) update;
        const uint32 vectors = (width * sizeof(uint32_t)) / hvx_bytes();
        for (uint32 r = 0; r < repeats; ++r) {
            for (uint32 slot = 0; slot < slots; ++slot) {
                const HVX_Vector delta = Q6_V_vsplat_R((int) (slot + (r & 31u)));
                HVX_Vector * row = (HVX_Vector *) (kv + (size_t) slot * width);
                for (uint32 i = 0; i < vectors; ++i) {
                    HVX_Vector add = Q6_Vw_vadd_VwVw(uv[i], delta);
                    row[i] = Q6_Vw_vadd_VwVw(row[i], add);
                }
            }
        }
    }

    *checksum = checksum_u32(kv, total);
    aligned_free_128(update);
    aligned_free_128(kv);
    return AEE_SUCCESS;
}

int aisp_htp_minimal_run(remote_handle64 handle, uint32 scenario, uint32 mode, uint32 elements, uint32 repeats, uint64 * checksum) {
    struct aisp_htp_minimal_context * ctx = (struct aisp_htp_minimal_context *) handle;
    if (!ctx || ctx->magic != AISP_HTP_MINIMAL_MAGIC || !checksum || repeats == 0u) {
        return AEE_EBADPARM;
    }
    if (mode != AISP_HTP_MODE_BASELINE && mode != AISP_HTP_MODE_OPTIMIZED) {
        return AEE_EBADPARM;
    }

    switch (scenario) {
        case AISP_HTP_SCENARIO_HVX_TILE:
            return run_hvx_tile(mode, elements, repeats, checksum);
        case AISP_HTP_SCENARIO_PIPELINE_FUSION:
            return run_pipeline_fusion(mode, elements, repeats, checksum);
        case AISP_HTP_SCENARIO_COPY_VECTORIZED:
            return run_copy_vectorized(mode, elements, repeats, checksum);
        case AISP_HTP_SCENARIO_KV_BLOCK:
            return run_kv_block(mode, elements, repeats, checksum);
        default:
            return AEE_EBADPARM;
    }
}
