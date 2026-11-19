#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cmath>
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

int parse(std::string_view token, int upper) {
    int value = 0;
    auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || value < 1 || value > upper)
        throw std::invalid_argument("invalid dimension or iteration count");
    return value;
}

std::vector<float> input(size_t size, uint32_t state) {
    std::vector<float> data(size);
    for (auto& item : data) {
        state = 1664525u * state + 1013904223u;
        item = (static_cast<int>(state >> 8) - 8388608) * 0x1p-23f;
    }
    return data;
}

int main(int argc, char** argv) try {
    if (argc != 7) throw std::invalid_argument("usage: gemm fp32|tf32 verify|sample M N K iterations");
    const std::string_view precision(argv[1]), command(argv[2]);
    const bool fast = precision == "tf32";
    if (!fast && precision != "fp32") throw std::invalid_argument("invalid precision");
    if (command != "verify" && command != "sample") throw std::invalid_argument("invalid command");
    int m = parse(argv[3], 8192), n = parse(argv[4], 8192), k = parse(argv[5], 8192);
    int iterations = parse(argv[6], 30);
    if (command == "verify" && (m > 256 || n > 256 || k > 256))
        throw std::invalid_argument("verify dimensions must be at most 256");
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0));
    if (device.major != 10 || device.minor != 0 || std::string_view(device.name) != "NVIDIA B200")
        throw std::runtime_error("NVIDIA B200 required");

    auto a = input(size_t(m) * k, 2027);
    auto b = input(size_t(k) * n, 2028);
    size_t size = size_t(m) * n;
    constexpr float sentinel = 123456.75f;
    std::vector<float> c(size + 1, sentinel);
    float *da, *db, *dc;
    check(cudaMalloc(&da, a.size() * sizeof(float)));
    check(cudaMalloc(&db, b.size() * sizeof(float)));
    check(cudaMalloc(&dc, c.size() * sizeof(float)));
    check(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dc + size, &sentinel, sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    check(cublasCreate(&handle));
    int version = 0;
    check(cublasGetVersion(handle, &version));
    constexpr float one = 1.0f, zero = 0.0f;
    auto multiply = [&] {
        check(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &one, da, CUDA_R_32F, m, db, CUDA_R_32F, k,
                           &zero, dc, CUDA_R_32F, m,
                           fast ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F_PEDANTIC,
                           CUBLAS_GEMM_DEFAULT));
    };
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    for (int warmup = 0; warmup < 3; ++warmup) multiply();
    check(cudaEventRecord(start));
    for (int step = 0; step < iterations; ++step) multiply();
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));
    check(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (c.back() != sentinel) throw std::runtime_error("output guard changed");

    const size_t checks = command == "verify" ? size : std::min(size, size_t(64));
    double diff_square = 0, ref_square = 0, max_abs = 0;
    for (size_t sample = 0; sample < checks; ++sample) {
        size_t index = sample * size / checks;
        int row = index % m, column = index / m;
        double expected = 0, sum_abs = 0;
        for (int inner = 0; inner < k; ++inner) {
            double term = double(a[row + size_t(inner) * m]) * double(b[inner + size_t(column) * k]);
            expected += term;
            sum_abs += std::abs(term);
        }
        double error = std::abs(double(c[index]) - expected);
        const double tolerance = (fast ? 0.001 : 0.000001) * sum_abs + 0.00001;
        if (!std::isfinite(c[index]) || error > tolerance)
            throw std::runtime_error("CPU oracle tolerance exceeded");
        max_abs = std::max(max_abs, error);
        diff_square += error * error;
        ref_square += expected * expected;
    }

    std::printf("%.9g,%.9g,%.9g,%zu,%d\n", double(ms) / iterations, max_abs,
                std::sqrt(diff_square / std::max(ref_square, 1e-30)), checks, version);
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
