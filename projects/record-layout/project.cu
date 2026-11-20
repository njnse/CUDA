#include <cuda_runtime.h>

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

using Word = uint32_t;
constexpr unsigned fields = 8;
constexpr unsigned threads = 256;
constexpr Word guard = 0x5a5a5a5au;

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

unsigned parse(std::string_view token, unsigned limit) {
    unsigned value = 0;
    auto result = std::from_chars(token.data(), token.data() + token.size(), value);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size() ||
        value == 0 || value > limit) throw std::invalid_argument("invalid count");
    return value;
}

template<bool columns, bool all>
__global__ void project(const Word* input, Word* output, unsigned n) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    if constexpr (all) {
        Word sum = 0;
        #pragma unroll
        for (unsigned field = 0; field < fields; ++field) {
            unsigned index = columns ? field * n + row : row * fields + field;
            sum += (field + 1) * input[index];
        }
        output[row] = sum;
    } else {
        output[row] = input[columns ? 3 * n + row : row * fields + 3];
    }
}

int main(int argc, char** argv) try {
    if (argc != 5) throw std::invalid_argument("usage: project rows|columns one|all n repeats");
    const std::string_view layout(argv[1]), selection(argv[2]);
    if ((layout != "rows" && layout != "columns") ||
        (selection != "one" && selection != "all"))
        throw std::invalid_argument("invalid layout or selection");
    const unsigned n = parse(argv[3], (1u << 25) + 3), repeats = parse(argv[4], 30);
    const bool columns = layout == "columns", all = selection == "all";
    cudaDeviceProp gpu{};
    check(cudaGetDeviceProperties(&gpu, 0));
    if (gpu.major != 10 || gpu.minor != 0 || std::string_view(gpu.name) != "NVIDIA B200")
        throw std::runtime_error("NVIDIA B200 required");

    std::vector<Word> input(size_t(n) * fields);
    for (unsigned field = 0; field < fields; ++field)
        for (unsigned row = 0; row < n; ++row)
            input[columns ? field * n + row : row * fields + field] =
                2654435761u * row + 2246822519u * (field + 1u) + 2027u;
    Word* source;
    Word* target;
    check(cudaMalloc(&source, input.size() * sizeof(Word)));
    check(cudaMalloc(&target, (n + 1) * sizeof(Word)));
    check(cudaMemcpy(source, input.data(), input.size() * sizeof(Word), cudaMemcpyHostToDevice));
    check(cudaMemcpy(target + n, &guard, sizeof(guard), cudaMemcpyHostToDevice));

    const unsigned grid = (n + threads - 1) / threads;
    auto launch = [&] {
        if (columns && all) project<true, true><<<grid, threads>>>(source, target, n);
        if (columns && !all) project<true, false><<<grid, threads>>>(source, target, n);
        if (!columns && all) project<false, true><<<grid, threads>>>(source, target, n);
        if (!columns && !all) project<false, false><<<grid, threads>>>(source, target, n);
        check(cudaGetLastError());
    };
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    for (int step = 0; step < 3; ++step) launch();
    check(cudaEventRecord(start));
    for (unsigned step = 0; step < repeats; ++step) launch();
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));

    std::vector<Word> output(n + 1);
    check(cudaMemcpy(output.data(), target, output.size() * sizeof(Word), cudaMemcpyDeviceToHost));
    if (output.back() != guard) throw std::runtime_error("guard overwritten");
    uint64_t checksum = 0;
    // Evaluate the logical row without using either layout or the staged input buffers.
    for (unsigned row = 0; row < n; ++row) {
        uint64_t base = uint64_t(row) * 2654435761u + 2027u;
        uint64_t expected = all ? 36u * base + 204u * uint64_t(2246822519u)
                                : base + 4u * uint64_t(2246822519u);
        if (output[row] != static_cast<Word>(expected)) throw std::runtime_error("CPU oracle mismatch");
        checksum += output[row];
    }
    std::printf("%.9g,%llu\n", double(ms) / repeats, static_cast<unsigned long long>(checksum));
    check(cudaEventDestroy(stop));
    check(cudaEventDestroy(start));
    check(cudaFree(target));
    check(cudaFree(source));
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
