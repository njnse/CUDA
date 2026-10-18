#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace transfer_lab {
constexpr unsigned seed=2027;
constexpr std::size_t max_n=4194307;
struct Chunk {std::size_t offset,count; unsigned slot;};
inline void check_shape(std::size_t n,std::size_t chunk,unsigned slots){
    if(!n||n>max_n||!chunk||chunk>(1u<<20)||(slots!=1&&slots!=2))
        throw std::invalid_argument("require n=1..4194307, chunk=1..1048576, slots=1 or 2");
}
inline std::vector<Chunk> plan(std::size_t n,std::size_t chunk,unsigned slots){
    check_shape(n,chunk,slots);std::vector<Chunk> v;
    for(std::size_t offset=0;offset<n;offset+=chunk)
        v.push_back({offset,std::min(chunk,n-offset),static_cast<unsigned>(v.size()%slots)});
    return v;
}
inline std::vector<std::uint32_t> input(std::size_t n){
    if(!n||n>max_n)throw std::invalid_argument("invalid length");
    std::vector<std::uint32_t>x(n);
    std::uint32_t state=seed;
    for(auto&v:x){state=1664525u*state+1013904223u;v=state;}
    // Directed values exercise wraparound even for the smallest shape.
    x[0]=0xffffffffu;if(n>1)x[1]=0;if(n>2)x[2]=0x80000000u;
    return x;
}
inline std::vector<std::uint32_t> oracle(const std::vector<std::uint32_t>&x){
    if(x.empty())throw std::invalid_argument("empty input");
    std::vector<std::uint32_t>y(x.size());
    for(std::size_t i=0;i<x.size();++i)
        y[i]=static_cast<std::uint32_t>((static_cast<std::uint64_t>(x[i])*1664525+1013904223)&UINT64_C(0xffffffff));
    return y;
}
}
