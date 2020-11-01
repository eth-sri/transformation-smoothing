#include "domains/interval.h"
#include "utils/constants.h"
#include "utils/utilities.h"
#include <iostream>
#include <vector>
#include <cassert>

#pragma once

class PointD {

public:
  std::vector<float_t> x;

  PointD() {}
  PointD(std::vector<float_t> x) { this->x = x; }
    
  operator std::vector<float_t>() const { return x; }
  PointD operator + (const PointD& other) const;
};

class HyperBox {

public:
    std::vector<Interval> it;
    size_t dim;
    
    HyperBox() { dim = 0; }
    HyperBox(std::vector<Interval> intervals) { this->it = intervals; this-> dim = intervals.size(); }

    Interval& operator[](int i);
    std::vector<HyperBox> split(int k) const; // split HyperBox in smaller HyperBoxes, make k splits per dimension
  float area() const;

  float maxDim() const;
  bool anyDimBelow(const float delta) const;

};

  std::ostream& operator << (std::ostream& os, const PointD& pt);
  std::ostream& operator << (std::ostream& os, const HyperBox& box);

