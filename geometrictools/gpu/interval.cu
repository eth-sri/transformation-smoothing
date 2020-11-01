#include "gpu/interval.h"
#include <limits>
#include <cassert>
#include <cmath>

__forceinline__ __device__ IntervalGPU::IntervalGPU(double inf, double sup) {
  assert(inf <= sup + Constants::EPS);
  this->inf = inf;
  this->sup = sup;
}

__forceinline__ __device__ IntervalGPU& IntervalGPU::operator += (const IntervalGPU &other) {
  this->inf += other.inf;
    this->sup += other.sup;
    return (*this);
}

__forceinline__ __device__ IntervalGPU::IntervalGPU() {
  this->inf = numeric_limits<double>::infinity();
    this->sup = -numeric_limits<double>::infinity();
}

__forceinline__ __device__ IntervalGPU IntervalGPU::getR() {
  return {-numeric_limits<double>::infinity(),
          numeric_limits<double>::infinity()};
}

__forceinline__ __device__ bool IntervalGPU::is_empty() const {
  return this->inf == numeric_limits<double>::infinity() &&
           this->sup == -numeric_limits<double>::infinity();
}

__forceinline__ __device__ IntervalGPU abs(const IntervalGPU& it) {
  if (it.sup < 0) {
    return IntervalGPU(-it.sup, -it.inf);
  } else if (it.inf > 0) {
    return IntervalGPU(it.inf, it.sup);
  }
  return {0, max(-it.inf, it.sup)};
}

__forceinline__ __device__ IntervalGPU normalizeAngle(IntervalGPU phi) {
  IntervalGPU ret = phi;
  while (ret.inf > M_PI) {
    ret = ret + (-2 * M_PI);
  }
  while (ret.inf < -M_PI) {
    ret = ret + (2 * M_PI);
  }
  return ret;
}

__forceinline__ __device__ bool IntervalGPU::contains(double x) const {
  return x >= inf - Constants::EPS && x <= sup + Constants::EPS;
}

__forceinline__ __device__ IntervalGPU IntervalGPU::cosine() const {
  if (sup - inf >= 2*M_PI) {
    return {-1, 1};
  }
  auto it = normalizeAngle(*this);
  assert(-M_PI <= it.inf and it.inf <= M_PI);

  double ret_inf = cos(it.inf);
  double ret_sup = cos(it.sup);
  IntervalGPU ret = IntervalGPU(min(ret_inf, ret_sup), max(ret_inf, ret_sup));

  if (it.contains(M_PI)) {
    ret.inf = -1;
  }
  if (it.contains(0) || it.contains(2*M_PI)) {
    ret.sup = 1;
  }

  return ret;
}

__forceinline__ __device__ IntervalGPU IntervalGPU::sine() const {
  if (sup - inf >= 2*M_PI) {
    return {-1, 1};
  }
  auto it = normalizeAngle(*this);
  assert(-M_PI <= it.inf and it.inf <= M_PI);
  assert(-M_PI <= it.sup and it.sup <= 3*M_PI);

  double ret_inf = sin(it.inf);
  double ret_sup = sin(it.sup);
  IntervalGPU ret = IntervalGPU(min(ret_inf, ret_sup), max(ret_inf, ret_sup));

  if (it.contains(-0.5*M_PI) || it.contains(1.5*M_PI)) {
    ret.inf = -1;
  }
  if (it.contains(0.5*M_PI) || it.contains(2.5*M_PI)){
    ret.sup = 1;
  }

  return ret;
}

__forceinline__ __device__ IntervalGPU IntervalGPU::square() const {
  if (inf > 0) {
    return {inf*inf, sup*sup};
  } else if (sup < 0) {
    return {sup*sup, inf*inf};
  } else {
    return {0, max(inf*inf, sup*sup)};
  }
}

// IntervalGPU IntervalGPU::sqrt() const {
//   assert(inf >= 0);
//   return {sqrt(inf), sqrt(sup)};
// }

__forceinline__ __device__ IntervalGPU IntervalGPU::operator - () const {
  return IntervalGPU(-sup, -inf);
}

__forceinline__ __device__ IntervalGPU IntervalGPU::operator + (const IntervalGPU& other) const {
  return IntervalGPU(inf + other.inf, sup + other.sup);
}

__forceinline__ __device__ IntervalGPU IntervalGPU::operator + (double other) const {
  return IntervalGPU(inf + other, sup + other);
}

__forceinline__ __device__ IntervalGPU IntervalGPU::operator - (double other) const {
  return *this + (-other);
}

__forceinline__ __device__ IntervalGPU IntervalGPU::operator - (const IntervalGPU& other) const {
  return -other + *this;
}

__forceinline__ __device__ IntervalGPU IntervalGPU::operator * (double other) const {
  if (other > 0) {
    return IntervalGPU(inf * other, sup * other);
  }
  return {sup * other, inf * other};
}

// double IntervalGPU::length() const {
//     return sup - inf;
// }

__forceinline__ __device__ IntervalGPU IntervalGPU::operator * (const IntervalGPU& other) const {
  IntervalGPU tmp1 = (*this) * other.inf;
    IntervalGPU tmp2 = (*this) * other.sup;
    return {min(tmp1.inf, tmp2.inf), max(tmp1.sup, tmp2.sup)};
}

__forceinline__ __device__ IntervalGPU IntervalGPU::meet(const IntervalGPU& other) const {
  if (this->is_empty() || other.is_empty()) {
    return IntervalGPU();
  }
  if (inf > other.sup || other.inf > sup) {
    return IntervalGPU();
  }
  return IntervalGPU(max(inf, other.inf), min(sup, other.sup));
}

__forceinline__ __device__ IntervalGPU IntervalGPU::join(const IntervalGPU& other) const {
  return IntervalGPU(min(inf, other.inf), max(sup, other.sup));
}

__forceinline__ __device__ IntervalGPU operator - (const double& a, const IntervalGPU &it) {
  return -it + a;
}

__forceinline__ __device__ IntervalGPU operator + (const double& a, const IntervalGPU &it) {
  return it + a;
}

__forceinline__ __device__ IntervalGPU operator * (const double& a, const IntervalGPU &it) {
  return it * a;
}

__forceinline__ __device__ IntervalGPU cos(IntervalGPU phi) {
  return phi.cosine();
}

__forceinline__ __device__ IntervalGPU sin(IntervalGPU phi) {
  return phi.sine();
}

__forceinline__ __device__ IntervalGPU IntervalGPU::pow(int k) const {
  if (k == 0) {
    return {1, 1};
  }
  IntervalGPU ret = {1, 1};
  for (int j = 0; j < k; ++j) {
    ret = ret * (*this);
  }
  return ret;
}
