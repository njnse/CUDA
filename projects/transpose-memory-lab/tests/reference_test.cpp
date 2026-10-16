#include "reference.hpp"
#include <iostream>
#include <utility>

int main() {
    if (transpose_reference({1,2,3,4,5,6}, 2, 3) !=
        std::vector<float>({1,4,2,5,3,6})) return 1;
    for (auto shape : {std::pair<int,int>{1,1}, {1,65}, {65,1}, {31,33},
                       {33,31}, {65,97}, {257,513}}) {
        int r = shape.first, c = shape.second;
        std::vector<float> x(size_t(r)*c);
        for (size_t i=0; i<x.size(); ++i) x[i] = float(i);
        auto y = transpose_reference(x,r,c);
        if (transpose_reference(y,c,r) != x) return 2;
    }
    bool rejected = false;
    try { transpose_reference({1},0,1); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 3;
    rejected = false;
    try { transpose_reference({1},2,2); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 4;
    std::cout << "PASS: known-answer, rectangular round-trip and invalid-input tests\n";
}
