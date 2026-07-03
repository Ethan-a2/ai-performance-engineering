#include "aisp_htp_minimal.h"

#include <AEEStdErr.h>
#include <remote.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

enum Scenario : uint32_t {
    SCENARIO_HVX_TILE = 0,
    SCENARIO_PIPELINE_FUSION = 1,
    SCENARIO_COPY_VECTORIZED = 2,
    SCENARIO_KV_BLOCK = 3,
};

uint32_t parse_scenario(const char * value) {
    if (std::strcmp(value, "hvx_tile") == 0) {
        return SCENARIO_HVX_TILE;
    }
    if (std::strcmp(value, "pipeline_fusion") == 0) {
        return SCENARIO_PIPELINE_FUSION;
    }
    if (std::strcmp(value, "copy_vectorized") == 0) {
        return SCENARIO_COPY_VECTORIZED;
    }
    if (std::strcmp(value, "kv_block") == 0) {
        return SCENARIO_KV_BLOCK;
    }
    throw std::runtime_error(std::string("unknown HTP scenario: ") + value);
}

uint32_t map_arch_capability(uint32_t capability) {
    switch (capability & 0xffU) {
        case 0x68: return 68;
        case 0x69: return 69;
        case 0x73: return 73;
        case 0x75: return 75;
        case 0x79: return 79;
        case 0x81: return 81;
        default: return 0;
    }
}

uint32_t forced_arch() {
    const char * value = std::getenv("AISP_HTP_MINIMAL_ARCH");
    if (!value || !*value) {
        return 0;
    }
    if (value[0] == 'v' || value[0] == 'V') {
        ++value;
    }
    return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
}

uint32_t query_htp_arch() {
    if (uint32_t arch = forced_arch()) {
        return arch;
    }

    remote_dsp_capability arch_ver{};
    arch_ver.domain = CDSP_DOMAIN_ID;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability = 0;

    const int err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, sizeof(arch_ver));
    if (err != AEE_SUCCESS) {
        std::fprintf(stderr, "failed to query HTP architecture via FastRPC: 0x%x\n", err);
        std::exit(2);
    }
    const uint32_t arch = map_arch_capability(arch_ver.capability);
    if (arch == 0) {
        std::fprintf(stderr, "unsupported HTP architecture capability: 0x%x\n", arch_ver.capability);
        std::exit(2);
    }
    return arch;
}

void enable_unsigned_pd() {
    remote_rpc_control_unsigned_module control{};
    control.domain = CDSP_DOMAIN_ID;
    control.enable = 1;
    const int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &control, sizeof(control));
    if (err != AEE_SUCCESS) {
        std::fprintf(stderr, "failed to enable unsigned CDSP PD: 0x%x\n", err);
        std::exit(2);
    }
}

void check_aee(int err, const char * what) {
    if (err != AEE_SUCCESS) {
        std::fprintf(stderr, "%s failed: 0x%x\n", what, err);
        std::exit(3);
    }
}

double time_mode(remote_handle64 handle, uint32_t scenario, uint32_t mode, uint32_t elements,
                 uint32_t repeats, uint32_t iterations, uint64 & checksum) {
    constexpr uint32_t warmup = 1;
    for (uint32_t i = 0; i < warmup; ++i) {
        uint64 ignored = 0;
        check_aee(aisp_htp_minimal_run(handle, scenario, mode, elements, repeats, &ignored), "warmup run");
    }

    const auto start = std::chrono::steady_clock::now();
    uint64 last_checksum = 0;
    for (uint32_t i = 0; i < iterations; ++i) {
        check_aee(aisp_htp_minimal_run(handle, scenario, mode, elements, repeats, &last_checksum), "benchmark run");
    }
    const auto end = std::chrono::steady_clock::now();
    checksum = last_checksum;
    return std::chrono::duration<double, std::milli>(end - start).count() / double(iterations);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 5) {
        std::fprintf(stderr, "usage: %s <hvx_tile|pipeline_fusion|copy_vectorized|kv_block> [iterations] [repeats] [elements]\n", argv[0]);
        return 2;
    }

    try {
        const char * scenario_name = argv[1];
        const uint32_t scenario = parse_scenario(scenario_name);
        const uint32_t iterations = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 8;
        const uint32_t repeats = argc > 3 ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 128;
        const uint32_t elements = argc > 4 ? static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10)) : 65536;

        const uint32_t arch = query_htp_arch();
        enable_unsigned_pd();

        char uri[256];
        std::snprintf(uri, sizeof(uri), "file:///libaisp_htp_minimal-v%u.so?aisp_htp_minimal_skel_handle_invoke&_modver=1.0%s", arch, CDSP_DOMAIN);

        remote_handle64 handle = -1;
        check_aee(aisp_htp_minimal_open(uri, &handle), "aisp_htp_minimal_open");

        uint32_t hw_threads = 0;
        uint32_t hvx_bytes = 0;
        uint32_t dsp_arch = 0;
        check_aee(aisp_htp_minimal_hwinfo(handle, &hw_threads, &hvx_bytes, &dsp_arch), "aisp_htp_minimal_hwinfo");

        uint64 baseline_checksum = 0;
        uint64 optimized_checksum = 0;
        const double baseline_ms = time_mode(handle, scenario, 0, elements, repeats, iterations, baseline_checksum);
        const double optimized_ms = time_mode(handle, scenario, 1, elements, repeats, iterations, optimized_checksum);

        check_aee(aisp_htp_minimal_close(handle), "aisp_htp_minimal_close");

        const int checksum_ok = baseline_checksum == optimized_checksum ? 1 : 0;
        const double speedup = optimized_ms > 0.0 ? baseline_ms / optimized_ms : 0.0;

        std::printf("device=hexagon_htp\n");
        std::printf("scenario=%s\n", scenario_name);
        std::printf("arch=v%u\n", arch);
        std::printf("dsp_arch=v%u\n", dsp_arch);
        std::printf("hw_threads=%u\n", hw_threads);
        std::printf("hvx_bytes=%u\n", hvx_bytes);
        std::printf("iterations=%u\n", iterations);
        std::printf("repeats=%u\n", repeats);
        std::printf("elements=%u\n", elements);
        std::printf("baseline_ms=%.9f\n", baseline_ms);
        std::printf("optimized_ms=%.9f\n", optimized_ms);
        std::printf("speedup=%.9f\n", speedup);
        std::printf("baseline_checksum=%llu\n", static_cast<unsigned long long>(baseline_checksum));
        std::printf("optimized_checksum=%llu\n", static_cast<unsigned long long>(optimized_checksum));
        std::printf("optimized_max_abs_err=%d\n", checksum_ok ? 0 : 1);

        return checksum_ok ? 0 : 4;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "%s\n", exc.what());
        return 2;
    }
}
