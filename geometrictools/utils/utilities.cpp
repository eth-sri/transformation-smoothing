#include "utils/utilities.h"


Coordinate<float> Rotation(const Coordinate<float>& coord, const float& phi) {
  return {coord.x * cos(phi) - coord.y * sin(phi),
    coord.x * sin(phi) + coord.y * cos(phi), coord.channel};
}

Coordinate<Interval> Rotation(const Coordinate<float> &coord, const Interval& phi) {
  return {coord.x * cos(phi) - coord.y * sin(phi),
    coord.x * sin(phi) + coord.y * cos(phi), coord.channel};
}

Coordinate<Interval> Rotation(const Coordinate<Interval> &coord, const Interval& phi) {
  return {coord.x * cos(phi) - coord.y * sin(phi),
    coord.x * sin(phi) + coord.y * cos(phi), coord.channel};
}


//factor 2 here is to obtain the correct translation in row/col coordinates
Coordinate<float> Translation(const Coordinate<float>& coord, const float dx, const float dy) {
  return {coord.x + 2*dx, coord.y + 2*dy, coord.channel};
}

Coordinate<Interval> Translation(const Coordinate<float> &coord, const Interval& dx, const Interval& dy) {
  return {coord.x + 2*dx, coord.y + 2*dy, coord.channel};
}

Coordinate<Interval> Translation(const Coordinate<Interval> &coord, const Interval& dx, const Interval& dy) {
  return {coord.x + 2*dx, coord.y + 2*dy, coord.channel};
}


void getGaussianFilter(float *G, const size_t filter_size, const float sigma) {
  float sum = 0;  
  const size_t c = filter_size / 2;
  for (size_t i = 0; i <= c; ++i) {
    for (size_t j = 0; j <= c; ++j) {
      const size_t r = i * i + j * j;
      const float f = (exp(-1.0f * r / sigma)) / (M_PI * sigma);
      G[i * (c+1) + j] = f;
      sum += 4 * f;
    }
  }
  for (size_t i = 0; i < (c+1)*(c+1); ++i) { 
    G[i] /= sum;
  }
}


float fadd_rd(const float& a, const float& b) {
  auto current = std::fegetround();
  std::fesetround(FE_DOWNWARD);
  float result = a + b;
  std::fesetround(current);
  return result;
}


float fadd_ru(const float& a, const float& b) {
  auto current = std::fegetround();
  std::fesetround(FE_UPWARD);
  float result = a + b;
  std::fesetround(current);
  return result;
}

float fmul_rd(const float& a, const float& b) {
  auto current = std::fegetround();
  std::fesetround(FE_DOWNWARD);
  float result = a * b;
  std::fesetround(current);
  return result;
}


float fmul_ru(const float& a, const float& b) {
  auto current = std::fegetround();
  std::fesetround(FE_UPWARD);
  float result = a * b;
  std::fesetround(current);
  return result;
}
