#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

using Word = uint32_t;
constexpr int tile_w = 32;
constexpr int tile_h = 8;
constexpr Word guard = 0x5a5a5a5au;

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

int parse(std::string_view token, int max) {
    int value = 0;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size() ||
        value < 1 || value > max) throw std::invalid_argument("invalid dimension or repeat count");
    return value;
}

__device__ Word read(const Word* data, int x, int y, int width, int height) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    return data[y * width + x];
}

__global__ void direct(const Word* input, Word* output, int width, int height) {
    int x = blockIdx.x * tile_w + threadIdx.x;
    int y = blockIdx.y * tile_h + threadIdx.y;
    if (x >= width || y >= height) return;
    output[y * width + x] = 4u * input[y * width + x] +
        read(input, x - 1, y, width, height) + read(input, x + 1, y, width, height) +
        read(input, x, y - 1, width, height) + read(input, x, y + 1, width, height);
}

__global__ void tiled(const Word* input, Word* output, int width, int height) {
    __shared__ Word tile[tile_h + 2][tile_w + 2];
    int lx = threadIdx.x, ly = threadIdx.y;
    int x = blockIdx.x * tile_w + lx;
    int y = blockIdx.y * tile_h + ly;
    tile[ly + 1][lx + 1] = read(input, x, y, width, height);
    if (lx == 0) {
        tile[ly + 1][0] = read(input, x - 1, y, width, height);
        tile[ly + 1][tile_w + 1] = read(input, x + tile_w, y, width, height);
    }
    if (ly == 0) {
        tile[0][lx + 1] = read(input, x, y - 1, width, height);
        tile[tile_h + 1][lx + 1] = read(input, x, y + tile_h, width, height);
    }
    __syncthreads();
    if (x < width && y < height)
        output[y * width + x] = 4u * tile[ly + 1][lx + 1] +
            tile[ly + 1][lx] + tile[ly + 1][lx + 2] + tile[ly][lx + 1] + tile[ly + 2][lx + 1];
}

int main(int argc, char** argv) try {
    if (argc != 5) throw std::invalid_argument("usage: stencil direct|tiled width height repeats");
    const std::string_view mode(argv[1]);
    if (mode != "direct" && mode != "tiled") throw std::invalid_argument("invalid mode");
    const int w = parse(argv[2], 8193), h = parse(argv[3], 8193), repeats = parse(argv[4], 30);
    cudaDeviceProp gpu{};
    check(cudaGetDeviceProperties(&gpu, 0));
    if (gpu.major != 10 || gpu.minor != 0 || std::string_view(gpu.name) != "NVIDIA B200")
        throw std::runtime_error("NVIDIA B200 required");

    const size_t size = size_t(w) * h;
    std::vector<Word> input(size);
    uint32_t state = 2027;
    for (auto& value : input) {
        state = 1664525u * state + 1013904223u;
        value = state;
    }
    Word* source;
    Word* target;
    check(cudaMalloc(&source, size * sizeof(Word)));
    check(cudaMalloc(&target, (size + 1) * sizeof(Word)));
    check(cudaMemcpy(source, input.data(), size * sizeof(Word), cudaMemcpyHostToDevice));
    check(cudaMemcpy(target + size, &guard, sizeof(guard), cudaMemcpyHostToDevice));

    const dim3 block(tile_w, tile_h);
    const dim3 grid((w + tile_w - 1) / tile_w, (h + tile_h - 1) / tile_h);
    auto launch = [&] {
        if (mode == "direct") direct<<<grid, block>>>(source, target, w, h);
        else tiled<<<grid, block>>>(source, target, w, h);
        check(cudaGetLastError());
    };
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    for (int step = 0; step < 3; ++step) launch();
    check(cudaEventRecord(start));
    for (int step = 0; step < repeats; ++step) launch();
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));

    std::vector<Word> output(size + 1);
    check(cudaMemcpy(output.data(), target, output.size() * sizeof(Word), cudaMemcpyDeviceToHost));
    if (output.back() != guard) throw std::runtime_error("output guard changed");
    uint64_t checksum = 0;
    for (size_t i = 0; i < size; ++i) {
        size_t row = i / w, column = i % w;
        size_t neighbors[] = {
            column ? i - 1 : i, column + 1 < size_t(w) ? i + 1 : i,
            row ? i - w : i, row + 1 < size_t(h) ? i + w : i
        };
        uint64_t expected = uint64_t(input[i]) * 4;
        for (size_t neighbor : neighbors) expected += input[neighbor];
        if (output[i] != expected % (1ull << 32)) throw std::runtime_error("CPU oracle mismatch");
        checksum += output[i];
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
