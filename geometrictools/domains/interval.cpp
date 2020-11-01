#include "domains/interval.h"
#include "utils/constants.h"
#include "utils/utilities.h"
#include <limits>
#include <iostream>
#include <cassert>
#include <cmath>

Interval::Interval(float inf, float sup) {
  if (!(inf <= sup))
    std::cout << inf << " " << sup << std::endl;
  assert(inf <= sup);
  this->inf = inf;
  this->sup = sup;
}

Interval& Interval::operator += (const Interval &other) {
  this->inf = fadd_rd(this->inf, other.inf);
  this->sup = fadd_ru(this->sup, other.sup);
  return (*this);
}

Interval::Interval() {
    this->inf = std::numeric_limits<float>::infinity();
    this->sup = -std::numeric_limits<float>::infinity();
}

Interval Interval::getR() {
    return {-std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
}

bool Interval::is_empty() const {
    return this->inf == std::numeric_limits<float>::infinity() &&
           this->sup == -std::numeric_limits<float>::infinity();
}

Interval abs(const Interval& it) {
  if (it.sup < 0) {
    return Interval(-it.sup, -it.inf);
  } else if (it.inf > 0) {
    return Interval(it.inf, it.sup);
  }
  return {0, std::max(-it.inf, it.sup)};
}

Interval normalizeAngle(Interval phi) {
  int k = 0;

  const float fmpi = (float) M_PI;
  
  if (phi.inf >= 0) {
    k = floor((phi.inf + fmpi) / (2 * fmpi));
  } else {
    k = ceil((phi.inf - fmpi) / (2 * fmpi));
  }

  float l = phi.inf - 2*M_PI*k;
  float u = phi.sup - 2*M_PI*k;
  l = std::max(-fmpi, l);
  u = std::max(-fmpi, u);
  l = std::min(fmpi, l);
  u = std::min(fmpi, u);  
  return Interval(l, u);
}
bool Interval::contains(float x) const {
  return x >= inf && x <= sup;
}

bool Interval::contains(const Interval& x) const {
  return x.inf >= inf && x.sup <= sup;
}

Interval Interval::cosine() const {
  const float fmpi = (float) M_PI;
  if (sup - inf >= 2*fmpi) {
    return {-1, 1};
  }
  auto it = normalizeAngle(*this);
  assert(-fmpi <= it.inf && it.inf <= fmpi);

  float ret_inf = cos(it.inf);
  float ret_sup = cos(it.sup);
  Interval ret = Interval(std::min(ret_inf, ret_sup), std::max(ret_inf, ret_sup));

  if (it.contains(M_PI)) {
    ret.inf = -1;
  }
  if (it.contains(0) || it.contains(2*M_PI)) {
    ret.sup = 1;
  }

  return ret;
}

Interval Interval::sine() const {
  const float fmpi = (float) M_PI;
  if (sup - inf >= 2*fmpi) {
    return {-1, 1};
  }
  auto it = normalizeAngle(*this);
  assert(-fmpi <= it.inf && it.inf <= fmpi);
  assert(-fmpi <= it.sup && it.sup <= 3*fmpi);

  float ret_inf = sin(it.inf);
  float ret_sup = sin(it.sup);
  Interval ret = Interval(std::min(ret_inf, ret_sup), std::max(ret_inf, ret_sup));

  if (it.contains(-0.5f*M_PI) || it.contains(1.5f*M_PI)) {
    ret.inf = -1;
  }
  if (it.contains(0.5f*M_PI) || it.contains(2.5f*M_PI)){
    ret.sup = 1;
  }

  return ret;
}

Interval Interval::square() const {
  if (inf > 0) {
    return {fmul_rd(inf,inf), fmul_ru(sup,sup)};
  } else if (sup < 0) {
    return {fmul_rd(sup,sup), fmul_ru(inf,inf)};
  } else {
    return {0, std::max(fmul_ru(inf,inf), fmul_ru(sup,sup))};
  }
}

Interval Interval::operator - () const {
  return Interval(-sup, -inf);
}

Interval Interval::operator + (const Interval& other) const {
  return Interval(fadd_rd(inf, other.inf), fadd_ru(sup, other.sup));
}

Interval Interval::operator + (float other) const {
  return Interval(fadd_rd(inf, other), fadd_ru(sup, other));
}

Interval Interval::operator - (float other) const {
  return *this + (-other);
}

Interval Interval::operator - (const Interval& other) const {
  return -other + *this;
}

Interval Interval::operator * (float other) const {
  if (other > 0) {
    return Interval(fmul_rd(inf, other), fmul_ru(sup, other));
  }
  return {fmul_rd(sup, other), fmul_ru(inf, other)};
}

Interval Interval::operator / (float other) const {
  assert(other != 0);
  return (1.0f / other) * *this;
}

// float Interval::length() const {
//     return sup - inf;
// }

Interval Interval::operator * (const Interval& other) const {
  Interval tmp1 = (*this) * other.inf;
  Interval tmp2 = (*this) * other.sup;
  return {std::min(tmp1.inf, tmp2.inf), std::max(tmp1.sup, tmp2.sup)};
}

Interval Interval::meet(const Interval& other) const {
  if (this->is_empty() || other.is_empty()) {
    return Interval();
  }
  if (inf > other.sup || other.inf > sup) {
    return Interval();
  }
  Interval out =  Interval(std::max(inf, other.inf), std::min(sup, other.sup));
  return out;
}

Interval Interval::join(const Interval& other) const {
  if (this->is_empty() && other.is_empty()) return Interval();
  else if (other.is_empty()) return *this;
  else if (this->is_empty()) return Interval(other);
  return Interval(std::min(inf, other.inf), std::max(sup, other.sup));
}

Interval operator - (const float& a, const Interval &it) {
  return -it + a;
}

Interval operator + (const float& a, const Interval &it) {
  return it + a;
}

Interval operator * (const float& a, const Interval &it) {
  return it * a;
}

std::ostream& operator<<(std::ostream& os, const Interval& it) {
  return os << "[" << it.inf << ", " << it.sup << "]";
}

Interval cos(Interval phi) {
  return phi.cosine();
}

Interval sin(Interval phi) {
  return phi.sine();
}

Interval Interval::pow(size_t k) const {
  if (k == 0) {
    return {1, 1};
  }
  else if (k % 2 == 1) {
    return {std::pow(inf, (float)k), std::pow(sup, (float)k)};
  } else {
    if (inf < 0 && sup > 0) {
      return {0, std::pow(std::max(-inf, sup), (float)k)};
    } else if (sup < 0) {
      return {std::pow(sup, (float)k), std::pow(inf, (float)k)};
    } else {
      return {std::pow(inf, (float)k), std::pow(sup, (float)k)};
    }
  }
}
