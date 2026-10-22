#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace geometry_lab {
constexpr unsigned seed=2027;
inline void check_shape(std::size_t n,unsigned rounds){
    if(!n||n>1048576||!rounds||rounds>64)throw std::invalid_argument("require length 1..1048576 and rounds 1..64");
}
struct Geometry {unsigned block,grid;};
inline Geometry geometry(std::size_t n,unsigned block,unsigned sms,unsigned max_threads,unsigned max_grid){
    check_shape(n,1);
    if(!block||block>max_threads||!sms||sms>std::numeric_limits<unsigned>::max()/32||!max_grid)
        throw std::invalid_argument("invalid launch limits");
    const auto needed=n/block+(n%block!=0);
    const auto grid=std::min<std::size_t>(needed,std::min<std::size_t>(32u*sms,max_grid));
    return {block,static_cast<unsigned>(grid)};
}
inline std::vector<std::uint32_t> input(std::size_t n){
    check_shape(n,1);std::vector<std::uint32_t>x(n);std::uint32_t state=seed;
    for(auto&v:x){state=1664525u*state+1013904223u;v=state;}
    x[0]=0xffffffffu;if(n>1)x[1]=0;if(n>2)x[2]=0x80000000u;return x;
}
inline std::vector<std::uint32_t> oracle(const std::vector<std::uint32_t>&x,unsigned rounds){
    check_shape(x.size(),rounds);
    // Compose a complete affine map in wide arithmetic; do not emulate GPU lanes.
    std::uint64_t a=1,b=0;
    for(unsigned k=0;k<rounds;++k){a=(1664525*a)&UINT64_C(0xffffffff);b=(1664525*b+1013904223)&UINT64_C(0xffffffff);}
    std::vector<std::uint32_t>y(x.size());
    for(std::size_t i=0;i<x.size();++i)y[i]=static_cast<std::uint32_t>((a*x[i]+b)&UINT64_C(0xffffffff));
    return y;
}
}
