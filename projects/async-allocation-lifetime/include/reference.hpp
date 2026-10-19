#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace pool_lab {
constexpr unsigned seed=2027;
inline void check_size(std::size_t n){if(!n||n>1048576)throw std::invalid_argument("length must be 1..1048576");}
inline std::vector<std::uint32_t> oracle(std::size_t n){
    check_size(n);std::vector<std::uint32_t> result(n);
    for(std::size_t i=0;i<n;++i)result[i]=static_cast<std::uint32_t>((i*UINT64_C(2654435761)+seed)&UINT64_C(0xffffffff));
    return result;
}
}
