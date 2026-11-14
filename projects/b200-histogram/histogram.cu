#include <cuda_runtime.h>

#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

constexpr unsigned bins = 256;
constexpr unsigned threads = 256;

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

__global__ void global_hist(const unsigned char* data, unsigned n, unsigned* counts) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(counts + data[i], 1);
}

__global__ void block_hist(const unsigned char* data, unsigned n, unsigned* counts) {
    __shared__ unsigned local[bins];
    local[threadIdx.x] = 0;
    __syncthreads();
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(local + data[i], 1);
    __syncthreads();
    if (local[threadIdx.x]) atomicAdd(counts + threadIdx.x, local[threadIdx.x]);
}

int main(int argc, char** argv) try {
    if (argc != 5) throw std::runtime_error("usage: histogram global|block uniform|hot N repetitions");
    const std::string_view mode(argv[1]), distribution(argv[2]);
    if ((mode != "global" && mode != "block") || (distribution != "uniform" && distribution != "hot"))
        throw std::runtime_error("unknown mode or distribution");
    auto parse = [](std::string_view input, unsigned min, unsigned max) {
        unsigned value = 0;
        auto result = std::from_chars(input.data(), input.data() + input.size(), value);
        if (result.ec != std::errc{} || result.ptr != input.end() || value < min || value > max)
            throw std::runtime_error("size or repetition count out of range");
        return value;
    };
    const unsigned n = parse(argv[3], 0, 1u << 27), repetitions = parse(argv[4], 1, 30);
    cudaDeviceProp gpu{};
    check(cudaGetDeviceProperties(&gpu, 0));
    if (gpu.major != 10 || gpu.minor != 0 || std::string_view(gpu.name).find("B200") == std::string_view::npos)
        throw std::runtime_error("NVIDIA B200 required");
    const unsigned grid = static_cast<unsigned>(gpu.multiProcessorCount) * 8u;
    std::vector<unsigned char> host(n);
    uint32_t seed = 2027;
    for (auto& value : host) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        value = distribution == "hot" && (seed & 15u) != 0 ? 7 : static_cast<unsigned char>(seed >> 24);
    }
    unsigned char* data = nullptr;
    unsigned* counts = nullptr;
    check(cudaMalloc(&data, n ? n : 1));
    check(cudaMalloc(&counts, (bins + 1) * sizeof(unsigned)));
    if (n) check(cudaMemcpy(data, host.data(), n, cudaMemcpyHostToDevice));
    constexpr unsigned guard = 0xAAAAAAAAu;
    check(cudaMemcpy(counts + bins, &guard, sizeof(guard), cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    double total = 0;
    for (unsigned run = 0; run < repetitions + 2; ++run) {
        check(cudaMemset(counts, 0, bins * sizeof(unsigned)));
        check(cudaEventRecord(start));
        if (mode == "global") global_hist<<<grid, threads>>>(data, n, counts);
        else block_hist<<<grid, threads>>>(data, n, counts);
        check(cudaGetLastError());
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, stop));
        if (run >= 2) total += ms;
    }
    std::array<unsigned, bins + 1> result{};
    check(cudaMemcpy(result.data(), counts, sizeof(result), cudaMemcpyDeviceToHost));
    if (result.back() != guard) throw std::runtime_error("output guard overwritten");
    check(cudaEventDestroy(stop));
    check(cudaEventDestroy(start));
    check(cudaFree(counts));
    check(cudaFree(data));
    std::printf("%.9f", total / repetitions);
    for (unsigned i = 0; i < bins; ++i) std::printf(",%u", result[i]);
    std::puts("");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
