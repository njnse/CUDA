#include <cuda_runtime.h>

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

using Word = uint32_t;
constexpr unsigned threads = 256, rounds = 24, key = 2027;
constexpr Word guard = 0xdd66dd66u;

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

unsigned parse(std::string_view token, unsigned lo, unsigned hi) {
    unsigned result = 0;
    auto parsed = std::from_chars(token.data(), token.data() + token.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() ||
        result < lo || result > hi) throw std::invalid_argument("invalid numeric argument");
    return result;
}

__global__ void transform(const Word* input, Word* target, unsigned n, Word salt) {
    extern __shared__ Word reserved[];
    if (threadIdx.x == 0) reserved[0] = salt;
    __syncthreads();
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Word value = input[i] ^ reserved[0];
        #pragma unroll 1
        for (unsigned step = 0; step < rounds; ++step)
            value = ((value << 7) | (value >> 25)) + 0x9e3779b9u + step;
        target[i] = value;
    }
}

int main(int argc, char** argv) try {
    if (argc != 2 && argc != 4)
        throw std::invalid_argument("usage: budget limits|reject or budget bytes elements repeats");
    std::string_view action(argv[1]);
    unsigned bytes = 0, n = 0, repeats = 0;
    if (argc == 4) {
        bytes = parse(action, 4, 1u << 20);
        n = parse(argv[2], 1, (1u << 23) + 3);
        repeats = parse(argv[3], 1, 30);
    } else if (action != "limits" && action != "reject") {
        throw std::invalid_argument("invalid action");
    }

    cudaDeviceProp gpu{};
    check(cudaGetDeviceProperties(&gpu, 0));
    if (gpu.major != 10 || gpu.minor != 0 || std::string_view(gpu.name) != "NVIDIA B200")
        throw std::runtime_error("NVIDIA B200 required");
    int optin = 0;
    check(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    cudaFuncAttributes attr{};
    check(cudaFuncGetAttributes(&attr, transform));
    const int allowed = optin - static_cast<int>(attr.sharedSizeBytes);
    if (action == "limits") {
        std::printf("{\"default_dynamic_bytes\":%d,\"optin_dynamic_bytes\":%d,"
                    "\"shared_bytes_per_sm\":%zu,\"threads_per_sm\":%d,\"registers_per_thread\":%d}\n",
                    attr.maxDynamicSharedSizeBytes, allowed, gpu.sharedMemPerMultiprocessor,
                    gpu.maxThreadsPerMultiProcessor, attr.numRegs);
        return 0;
    }
    if (action == "reject") {
        auto result = cudaFuncSetAttribute(transform, cudaFuncAttributeMaxDynamicSharedMemorySize, allowed + 1);
        if (result != cudaErrorInvalidValue) throw std::runtime_error("expected cudaErrorInvalidValue");
        std::puts("PASS opt-in above device limit rejected with cudaErrorInvalidValue");
        return 0;
    }
    if (bytes > static_cast<unsigned>(allowed)) throw std::invalid_argument("request exceeds opt-in limit");
    check(cudaFuncSetAttribute(transform, cudaFuncAttributeMaxDynamicSharedMemorySize, allowed));
    int blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, transform, threads, bytes));

    std::vector<Word> input(n), output(n + 1);
    for (unsigned i = 0; i < n; ++i) input[i] = 2654435761u * i + key;
    Word *source, *target;
    check(cudaMalloc(&source, n * sizeof(Word)));
    check(cudaMalloc(&target, (n + 1) * sizeof(Word)));
    check(cudaMemcpy(source, input.data(), n * sizeof(Word), cudaMemcpyHostToDevice));
    check(cudaMemcpy(target + n, &guard, sizeof(guard), cudaMemcpyHostToDevice));

    auto launch = [&] {
        transform<<<(n + threads - 1) / threads, threads, bytes>>>(source, target, n, key);
        check(cudaGetLastError());
    };
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    for (int i = 0; i < 3; ++i) launch();
    check(cudaEventRecord(start));
    for (unsigned i = 0; i < repeats; ++i) launch();
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start, stop));

    check(cudaMemcpy(output.data(), target, output.size() * sizeof(Word), cudaMemcpyDeviceToHost));
    if (output.back() != guard) throw std::runtime_error("output guard overwritten");
    constexpr uint64_t radix = 1ull << 32;
    uint64_t checksum = 0;
    for (unsigned i = 0; i < n; ++i) {
        uint64_t value = ((2654435761ull * i + key) % radix) ^ key;
        for (unsigned step = 0; step < rounds; ++step)
            value = (128 * value + value / (1u << 25) + 0x9e3779b9u + step) % radix;
        if (output[i] != value) throw std::runtime_error("CPU oracle mismatch");
        checksum += output[i];
    }

    std::printf("%.9g,%llu,%d\n", double(elapsed) / repeats,
                static_cast<unsigned long long>(checksum), blocks);
    check(cudaEventDestroy(stop));
    check(cudaEventDestroy(start));
    check(cudaFree(target));
    check(cudaFree(source));
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
