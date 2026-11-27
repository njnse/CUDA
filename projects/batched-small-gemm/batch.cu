#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(cublasGetStatusString(status));
}

int parse(std::string_view token, int maximum) {
    int value = 0;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() ||
        value < 1 || value > maximum) throw std::invalid_argument("invalid dimension, batch or repeat count");
    return value;
}

std::vector<float> input(size_t size, uint32_t seed) {
    std::vector<float> values(size);
    for (auto& value : values) {
        seed = 1664525u * seed + 1013904223u;
        value = static_cast<int>((seed >> 16) % 19u) - 9;
    }
    return values;
}

int main(int argc, char** argv) try {
    if (argc != 8) throw std::invalid_argument("usage: batch loop|strided verify|sample M N K count repeats");
    const std::string_view mode(argv[1]), command(argv[2]);
    if ((mode != "loop" && mode != "strided") || (command != "verify" && command != "sample"))
        throw std::invalid_argument("invalid mode or command");
    const int m = parse(argv[3], 128), n = parse(argv[4], 128), k = parse(argv[5], 128);
    const int count = parse(argv[6], 1024), repeats = parse(argv[7], 20);
    if (command == "verify" && count > 8) throw std::invalid_argument("verify batch must be at most eight");

    cudaDeviceProp gpu{};
    check(cudaGetDeviceProperties(&gpu, 0));
    if (gpu.major != 10 || gpu.minor != 0 || std::string_view(gpu.name) != "NVIDIA B200")
        throw std::runtime_error("NVIDIA B200 required");
    const size_t stride_a = size_t(m) * k, stride_b = size_t(k) * n, stride_c = size_t(m) * n;
    auto a = input(stride_a * count, 2027);
    auto b = input(stride_b * count, 2028);
    const size_t size = stride_c * count;
    constexpr float sentinel = 123456.75f;
    std::vector<float> c(size + 1, sentinel);
    float *da, *db, *dc;
    check(cudaMalloc(&da, a.size() * sizeof(float)));
    check(cudaMalloc(&db, b.size() * sizeof(float)));
    check(cudaMalloc(&dc, c.size() * sizeof(float)));
    check(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dc, c.data(), c.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    check(cublasCreate(&handle));
    check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    int version = 0;
    check(cublasGetVersion(handle, &version));
    constexpr float alpha = 1.0f, beta = 0.0f;
    auto step = [&] {
        if (mode == "strided") {
            check(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                                           da, m, stride_a, db, k, stride_b, &beta,
                                           dc, m, stride_c, count));
        } else {
            for (int item = 0; item < count; ++item)
                check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                                  da + item * stride_a, m, db + item * stride_b, k, &beta,
                                  dc + item * stride_c, m));
        }
    };
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    for (int warmup = 0; warmup < 3; ++warmup) step();
    check(cudaEventRecord(start));
    for (int repeat = 0; repeat < repeats; ++repeat) step();
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));

    check(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (c.back() != sentinel) throw std::runtime_error("output guard changed");
    const size_t probes = command == "verify" ? size : std::min(size, size_t(257));
    int64_t checksum = 0;
    for (size_t probe = 0; probe < probes; ++probe) {
        const size_t index = probes == 1 ? 0 : probe * (size - 1) / (probes - 1);
        const size_t item = index / stride_c, local = index % stride_c;
        const size_t row = local % m, column = local / m;
        int expected = 0;
        for (int t = 0; t < k; ++t)
            expected += static_cast<int>(a[item * stride_a + row + size_t(t) * m]) *
                        static_cast<int>(b[item * stride_b + t + column * k]);
        if (c[index] != expected) throw std::runtime_error("CPU oracle mismatch");
        checksum += expected;
    }
    std::printf("%.9g,%lld,%zu,%d\n", double(ms) / repeats, static_cast<long long>(checksum), probes, version);
    check(cudaEventDestroy(stop));
    check(cudaEventDestroy(start));
    check(cublasDestroy(handle));
    check(cudaFree(dc));
    check(cudaFree(db));
    check(cudaFree(da));
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
