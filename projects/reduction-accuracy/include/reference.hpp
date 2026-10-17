#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace reduction_lab {
constexpr unsigned seed=2027;
inline void check_size(std::size_t n) {
    if (!n || n>(1u<<20)) throw std::invalid_argument("length must be 1..1048576");
}
inline std::vector<float> input(std::size_t n,const std::string& dataset) {
    check_size(n);
    std::vector<float> x(n);
    std::uint32_t state=seed;
    for(std::size_t i=0;i<n;++i) {
        if(dataset=="positive") x[i]=1.0f+static_cast<float>(i%8)*0.125f;
        else if(dataset=="cancellation") {
            const float pattern[]={100000000.0f,1.0f,-100000000.0f,1.0f};
            x[i]=pattern[i%4];
        } else if(dataset=="dyadic") {
            state=1664525u*state+1013904223u;
            x[i]=static_cast<float>(static_cast<int>(state%2049)-1024)/16.0f;
        } else throw std::invalid_argument("unknown dataset");
    }
    return x;
}
struct Reference { long double sum, sum_abs; };
// Independent compensated long-double accumulation; no GPU tree emulation.
inline Reference oracle(const std::vector<float>& x) {
    if(x.empty()) throw std::invalid_argument("empty input");
    long double sum=0,correction=0,absolute=0;
    for(float f:x) {
        if(!std::isfinite(f)) throw std::invalid_argument("finite input required");
        const long double v=f,t=sum+v;
        correction += std::fabs(sum)>=std::fabs(v) ? (sum-t)+v : (v-t)+sum;
        sum=t; absolute+=std::fabs(v);
    }
    return {sum+correction,absolute};
}
inline long double gamma(std::size_t additions,long double unit) {
    const long double v=additions*unit;
    if(v>=1) throw std::invalid_argument("invalid error bound");
    return v/(1-v);
}
} // namespace reduction_lab
