#include "reference.hpp"
#include <iostream>
static void need(bool p){if(!p)throw std::runtime_error("CPU oracle failed");}
int main(){
    using namespace pool_lab;std::size_t checks=0;
    need(oracle(3)==std::vector<std::uint32_t>({2027u,2654437788u,1013906253u}));++checks;
    for(std::size_t n:{1u,31u,256u,257u,65537u,1048576u}){
        auto values=oracle(n);std::uint32_t x=seed;
        for(auto v:values){need(x==v);x+=2654435761u;}
        ++checks;
    }
    for(std::size_t n:{0u,1048577u}){bool rejected=false;try{check_size(n);}catch(const std::invalid_argument&){rejected=true;}need(rejected);++checks;}
    std::cout<<"PASS wide-integer oracle vs incremental wraparound, known answers, tails, invalid sizes: "<<checks<<" checks\n";
}
