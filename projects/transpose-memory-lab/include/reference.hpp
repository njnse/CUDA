#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

inline std::vector<float> transpose_reference(const std::vector<float>& input,
                                            int rows, int cols) {
    if (rows <= 0 || cols <= 0 || input.size() != size_t(rows) * size_t(cols))
        throw std::invalid_argument("positive dimensions and matching input required");
    std::vector<float> output(input.size());
    // Independent flat-index oracle, intentionally different from the tiled GPU mapping.
    for (size_t i = 0; i < output.size(); ++i)
        output[i] = input[(i % size_t(rows)) * size_t(cols) + i / size_t(rows)];
    return output;
}
