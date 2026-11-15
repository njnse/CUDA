#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

using Word = uint32_t;
constexpr unsigned block = 256;

void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}

__global__ void affine(const Word* in, Word* temp, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) temp[i] = 3u * in[i] + 7u;
}

__global__ void combine(const Word* temp, const Word* bias, Word* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (temp[i] ^ 0xAA55AA55u) + bias[i];
}

__global__ void fused(const Word* in, const Word* bias, Word* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ((3u * in[i] + 7u) ^ 0xAA55AA55u) + bias[i];
}

int main(int argc, char** argv) try {
    if (argc != 4) throw std::invalid_argument("usage: fusion split|fused size repeats");
    std::string_view mode(argv[1]);
    if (mode != "split" && mode != "fused") throw std::invalid_argument("invalid mode");
    auto parse = [](const char* value, unsigned limit) {
        if (*value < '0' || *value > '9') throw std::invalid_argument("invalid count");
        char* tail = nullptr;
        const auto count = std::strtoul(value, &tail, 10);
        if (*tail || !count || count > limit) throw std::invalid_argument("invalid count");
        return static_cast<unsigned>(count);
    };
    unsigned n = parse(argv[2], 1u << 26), repeats = parse(argv[3], 50);
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 10 || prop.minor != 0 || std::string_view(prop.name) != "NVIDIA B200")
        throw std::runtime_error("NVIDIA B200 required");
    Word *in, *bias, *out, *temp = nullptr;
    check(cudaMalloc(&in, n * sizeof(Word)));
    check(cudaMalloc(&bias, n * sizeof(Word)));
    check(cudaMalloc(&out, (n + 1) * sizeof(Word)));
    if (mode == "split") check(cudaMalloc(&temp, n * sizeof(Word)));
    auto* host = new Word[n + 1];
    for (unsigned i = 0; i < n; ++i) host[i] = i * 2654435761u + 2027u;
    check(cudaMemcpy(in, host, n * sizeof(Word), cudaMemcpyHostToDevice));
    for (unsigned i = 0; i < n; ++i) host[i] = (i ^ 17u) * 65537u;
    check(cudaMemcpy(bias, host, n * sizeof(Word), cudaMemcpyHostToDevice));
    check(cudaMemset(out, 0x5a, (n + 1) * sizeof(Word)));

    unsigned grid = (n + block - 1) / block;
    auto launch = [&] {
        if (mode == "split") {
            affine<<<grid, block>>>(in, temp, n);
            check(cudaGetLastError());
            combine<<<grid, block>>>(temp, bias, out, n);
        } else {
            fused<<<grid, block>>>(in, bias, out, n);
        }
        check(cudaGetLastError());
    };
    cudaEvent_t start, end;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&end));
    for (int i = 0; i < 3; ++i) launch();
    check(cudaEventRecord(start));
    for (unsigned i = 0; i < repeats; ++i) launch();
    check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start, end));

    check(cudaMemcpy(host, out, (n + 1) * sizeof(Word), cudaMemcpyDeviceToHost));
    if (host[n] != 0x5a5a5a5au) throw std::runtime_error("output guard overwritten");
    uint64_t sum = 0;
    for (unsigned i = 0; i < n; ++i) {
        const uint64_t input = (uint64_t(i) * 2654435761u + 2027u) % (1ull << 32);
        const uint64_t addend = (uint64_t(i ^ 17u) * 65537u) % (1ull << 32);
        const uint64_t first = (input + input + input + 7u) % (1ull << 32);
        const uint64_t expected = ((first ^ 0xAA55AA55u) + addend) % (1ull << 32);
        if (host[i] != expected) throw std::runtime_error("CPU oracle mismatch");
        sum += host[i];
    }
    std::printf("%.9g,%llu\n", elapsed / repeats, static_cast<unsigned long long>(sum));
    delete[] host;
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(end));
    if (temp) check(cudaFree(temp));
    check(cudaFree(out));
    check(cudaFree(bias));
    check(cudaFree(in));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
