#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

using Word = uint32_t;
constexpr unsigned block = 256;
constexpr Word guard = 0x5a5a5a5au;

void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}

__global__ void direct(const Word* input, unsigned n, Word threshold, Word* out, Word* count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && (input[i] >> 24) < threshold) out[atomicAdd(count, 1u)] = i;
}

__global__ void warp_group(const Word* input, unsigned n, Word threshold, Word* out, Word* count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x % 32;
    unsigned selected = __ballot_sync(0xffffffffu, i < n && (input[i] >> 24) < threshold);
    if (selected & (1u << lane)) {
        unsigned leader = __ffs(selected) - 1;
        unsigned base = lane == leader ? atomicAdd(count, __popc(selected)) : 0;
        base = __shfl_sync(selected, base, leader);
        unsigned rank = __popc(selected & ((1u << lane) - 1u));
        out[base + rank] = i;
    }
}

int main(int argc, char** argv) try {
    if (argc != 5) throw std::invalid_argument("invalid arguments: compact direct|warp n threshold repeats");
    std::string_view mode(argv[1]);
    if (mode != "direct" && mode != "warp") throw std::invalid_argument("invalid mode");
    auto parse = [](const char* token, unsigned lo, unsigned hi) {
        if (*token < '0' || *token > '9') throw std::invalid_argument("invalid count");
        char* end;
        auto value = std::strtoul(token, &end, 10);
        if (*end || value < lo || value > hi) throw std::invalid_argument("invalid count");
        return static_cast<unsigned>(value);
    };
    unsigned n = parse(argv[2], 1, 1u << 25), threshold = parse(argv[3], 0, 256);
    unsigned repeats = parse(argv[4], 1, 30);
    cudaDeviceProp gpu{};
    check(cudaGetDeviceProperties(&gpu, 0));
    if (gpu.major != 10 || gpu.minor != 0 || std::string_view(gpu.name) != "NVIDIA B200")
        throw std::runtime_error("NVIDIA B200 required");

    std::vector<Word> input(n), output(n + 1);
    for (unsigned i = 0; i < n; ++i) input[i] = i * 2654435761u + 2027u;
    Word *d_input, *d_output, *d_count;
    check(cudaMalloc(&d_input, n * sizeof(Word)));
    check(cudaMalloc(&d_output, (n + 1) * sizeof(Word)));
    check(cudaMalloc(&d_count, sizeof(Word)));
    check(cudaMemcpy(d_input, input.data(), n * sizeof(Word), cudaMemcpyHostToDevice));
    check(cudaMemset(d_output, 0x5a, (n + 1) * sizeof(Word)));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    double total = 0;
    unsigned grid = (n + block - 1) / block;
    for (unsigned sample = 0; sample < repeats + 3; ++sample) {
        check(cudaMemset(d_count, 0, sizeof(Word)));
        check(cudaEventRecord(start));
        if (mode == "direct") direct<<<grid, block>>>(d_input, n, threshold, d_output, d_count);
        else warp_group<<<grid, block>>>(d_input, n, threshold, d_output, d_count);
        check(cudaGetLastError());
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, stop));
        if (sample >= 3) total += ms;
    }
    Word count, tail;
    check(cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&tail, d_output + n, sizeof(tail), cudaMemcpyDeviceToHost));
    if (count > n || tail != guard) throw std::runtime_error("output extent mismatch");
    if (count) check(cudaMemcpy(output.data(), d_output, count * sizeof(Word), cudaMemcpyDeviceToHost));
    std::vector<bool> seen(n);
    for (unsigned pos = 0; pos < count; ++pos) {
        Word index = output[pos];
        if (index >= n || seen[index]) throw std::runtime_error("invalid or duplicate index");
        seen[index] = true;
    }
    for (unsigned i = 0; i < n; ++i) {
        uint64_t value = (uint64_t(i) * 2654435761u + 2027u) % (1ull << 32);
        if (seen[i] != (value / (1ull << 24) < threshold)) throw std::runtime_error("CPU oracle mismatch");
    }
    std::printf("%.9g,%u\n", total / repeats, count);
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(d_count));
    check(cudaFree(d_output));
    check(cudaFree(d_input));
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
