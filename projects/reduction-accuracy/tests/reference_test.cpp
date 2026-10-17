#include "reference.hpp"
#include <iostream>
#include <limits>
static void require(bool p) { if(!p) throw std::runtime_error("oracle test failed"); }
int main() {
    using namespace reduction_lab;
    require(oracle({1,2,3}).sum==6);
    require(oracle({100000000.0f,1,-100000000.0f,1}).sum==2);
    std::size_t checks=2;
    for(auto n:{1u,31u,255u,256u,257u,65537u,1048576u}) {
        for(const auto& d:{"positive","cancellation","dyadic"}) {
            auto x=input(n,d); auto r=oracle(x);
            // Independent integer units of 1/16 are exactly representable here.
            std::int64_t units=0;
            for(float v:x) units+=static_cast<std::int64_t>(static_cast<double>(v)*16);
            require(r.sum==static_cast<long double>(units)/16);
            require(r.sum_abs>=std::fabs(r.sum)); ++checks;
        }
    }
    for(int kind=0;kind<5;++kind) {
        bool rejected=false;
        try {
            if(kind==0) check_size(0);
            if(kind==1) check_size(1048577);
            if(kind==2) (void)input(1,"unknown");
            if(kind==3) (void)oracle({});
            if(kind==4) (void)oracle({std::numeric_limits<float>::infinity()});
        } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected); ++checks;
    }
    require(gamma(2,0.125L)==1.0L/3); ++checks;
    std::cout<<"PASS long-double oracle vs exact integer units, cancellation, tails, finite-input contract: "<<checks<<" checks\n";
}
