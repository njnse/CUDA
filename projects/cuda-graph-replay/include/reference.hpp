#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>
inline std::vector<float> reference(const std::vector<float>& x) {
    if(x.empty()) throw std::invalid_argument("input must not be empty");
    std::vector<float> y(x.size());
    for(size_t i=0;i<x.size();++i) y[i]=float(std::max(1.25*double(x[i])-0.5,0.0));
    return y;
}
