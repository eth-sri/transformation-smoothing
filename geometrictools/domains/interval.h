#include <ostream>
#include <vector>

#pragma once

class Interval {
 public:
  float inf;
  float sup;

  Interval(float inf, float sup);
  Interval();

  Interval(const Interval &other) : inf(other.inf), sup(other.sup) {};
  Interval& operator= (const Interval &other) {
    this->inf = other.inf;
    this->sup = other.sup;
    return *this;
  };
  
  bool is_empty() const;
  bool contains(float x) const;
  bool contains(const Interval& x) const;  
  Interval cosine() const;
  Interval sine() const;
  Interval square() const;
  Interval operator + (const Interval& other) const;
  Interval operator - (const Interval& other) const;
  Interval operator * (const Interval& other) const;
  Interval operator - () const;
  Interval& operator += (const Interval& other);


  Interval operator - (float other) const;
  Interval operator + (float other) const;
  Interval operator * (float other) const;
  Interval operator / (float other) const;

  Interval meet(const Interval& other) const;
  Interval join(const Interval& other) const;
  Interval pow(size_t k) const;

  static Interval join(std::vector<Interval> intervals);
  static Interval getR();
};

Interval operator - (const float& a, const Interval &it);
Interval operator + (const float& a, const Interval &it);
Interval operator * (const float& a, const Interval &it);
std::ostream& operator<<(std::ostream& os, const Interval& it);
Interval normalizeAngle(Interval phi);

Interval cos(Interval phi);
Interval sin(Interval phi);
Interval abs(const Interval& it);
