#include <stdint.h>

extern "C" __global__ void transform(const uint32_t* input, uint32_t* output,
                                     uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 3u * input[i] + (i % 17u);
}
