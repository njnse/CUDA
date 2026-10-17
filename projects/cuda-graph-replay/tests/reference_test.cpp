#include "reference.hpp"
#include <iostream>
int main() {
    if(reference({-2,0,0.25f,0.5f,4})!=std::vector<float>({0,0,0,0.125f,4.5f})) return 1;
    bool rejected=false;
    try {reference({});} catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected) return 2;
    auto y=reference(std::vector<float>(257,0.5f));
    for(float v:y) if(v!=0.125f) return 3;
    std::cout<<"PASS known-answer, tail length, empty rejection\n";
}
