#include "domains/interval.h"
#include <cmath>
#include <cfenv>

using namespace std;

#pragma once

#pragma STDC FENV_ACCESS ON

template <class T>
class Coordinate {

public:
  T x;
  T y;
  size_t channel;

  Coordinate(T x, T y, size_t channel) : x(x), y(y), channel(channel) {}
};

enum Transform {
  rotation=0,
  translation=1
};

Coordinate<float> Rotation(const Coordinate<float>& coord, const float& phi);
Coordinate<Interval> Rotation(const Coordinate<float> &coord, const Interval& phi);
Coordinate<Interval> Rotation(const Coordinate<Interval> &coord, const Interval& phi);


Coordinate<float> Translation(const Coordinate<float>& coord, const float dx, const float dy);
Coordinate<Interval> Translation(const Coordinate<float> &coord, const Interval& dx, const Interval& dy);
Coordinate<Interval> Translation(const Coordinate<Interval> &coord, const Interval& dx, const Interval& dy);


void getGaussianFilter(float *G, const size_t filter_size, const float sigma);


float fadd_rd(const float& a, const float& b);
float fadd_ru(const float& a, const float& b);
float fmul_rd(const float& a, const float& b);
float fmul_ru(const float& a, const float& b);
