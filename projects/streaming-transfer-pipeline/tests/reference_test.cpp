#include "reference.hpp"
#include <iostream>
static void need(bool p){if(!p)throw std::runtime_error("CPU contract failure");}
int main(){
    using namespace transfer_lab;
    need(oracle({0,1,0xffffffffu,0x80000000u})==std::vector<std::uint32_t>({1013904223u,1015568748u,1012239698u,3161387871u}));
    std::size_t checks=1;
    for(std::size_t n:{1u,31u,257u,65537u,1048579u,4194307u})for(std::size_t c:{256u,4096u,65536u})for(unsigned slots:{1u,2u}){
        auto chunks=plan(n,c,slots);std::size_t end=0;
        auto x=input(n),expected=oracle(x);std::vector<std::uint32_t>actual(n);
        for(std::size_t k=0;k<chunks.size();++k){auto q=chunks[k];
            need(q.offset==end&&q.count>0&&q.count<=c&&q.slot==k%slots);
            // Separate scalar modulo-32 arithmetic verifies the 64-bit oracle.
            for(std::size_t j=q.offset;j<q.offset+q.count;++j)actual[j]=x[j]*1664525u+1013904223u;
            end+=q.count;
        }
        need(end==n&&actual==expected);++checks;
    }
    for(int k=0;k<7;++k){bool rejected=false;
        try{
            if(k==0)check_shape(0,1,1);
            if(k==1)check_shape(max_n+1,1,1);
            if(k==2)check_shape(1,0,1);
            if(k==3)check_shape(1,1048577,1);
            if(k==4)check_shape(1,1,0);
            if(k==5)check_shape(1,1,3);
            if(k==6)(void)oracle({});
        }catch(const std::invalid_argument&){rejected=true;}
        need(rejected);++checks;
    }
    std::cout<<"PASS oracle known answers, wraparound, complete chunk coverage, tails, slot reuse, invalid contracts: "<<checks<<" checks\n";
}
