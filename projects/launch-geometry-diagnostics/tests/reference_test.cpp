#include "reference.hpp"
#include <iostream>
static void need(bool p){if(!p)throw std::runtime_error("CPU contract failure");}
int main(){
    using namespace geometry_lab;std::size_t checks=0;
    need(oracle({0,1,0xffffffffu},1)==std::vector<std::uint32_t>({1013904223u,1015568748u,1012239698u}));++checks;
    for(std::size_t n:{1u,31u,257u,65537u,1048576u}){
        auto x=input(n);
        for(unsigned rounds:{1u,2u,64u}){auto iterative=x;
            for(unsigned k=0;k<rounds;++k)for(auto&v:iterative)v=v*1664525u+1013904223u;
            need(iterative==oracle(x,rounds));++checks;
        }
        for(unsigned block:{128u,256u,512u,1024u}){
            auto launch=geometry(n,block,3,1024,2147483647u);std::vector<unsigned char>visits(n,0);
            for(std::size_t lane=0;lane<static_cast<std::size_t>(launch.grid)*launch.block;++lane)
                for(std::size_t i=lane;i<n;i+=static_cast<std::size_t>(launch.grid)*launch.block)++visits[i];
            need(std::all_of(visits.begin(),visits.end(),[](unsigned char v){return v==1;}));++checks;
        }
    }
    need(geometry(1048576,128,58,1024,10).grid==10);++checks;
    for(int k=0;k<10;++k){bool rejected=false;try{
        if(k==0)check_shape(0,1);
        if(k==1)check_shape(1048577,1);
        if(k==2)check_shape(1,0);
        if(k==3)check_shape(1,65);
        if(k==4)(void)geometry(1,0,1,1024,100);
        if(k==5)(void)geometry(1,1025,1,1024,100);
        if(k==6)(void)geometry(1,128,0,1024,100);
        if(k==7)(void)geometry(1,128,1,1024,0);
        if(k==8)(void)geometry(1,128,0xffffffffu,1024,100);
        if(k==9)(void)oracle({},1);
    }catch(const std::invalid_argument&){rejected=true;}need(rejected);++checks;}
    std::cout<<"PASS oracle, wraparound, exact grid-stride coverage, grid cap, invalid shapes and launch limits: "<<checks<<" checks\n";
}
