#include "utils/hyperbox.h"

PointD PointD::operator + (const PointD& other) const {
    assert(x.size() == other.x.size());
    std::vector<float_t> retX = this->x;
    for (size_t i = 0; i < other.x.size(); ++i) {
        retX[i] += other.x[i];
    }
    return PointD(retX);
}

Interval& HyperBox::operator[](int i) {
    return it[i];
}

float HyperBox::area() const {
  float area = 1;
  for (size_t i = 0; i < dim; ++i) {
    float_t size = (it[i].sup - it[i].inf);
    area *= size;
  }
  return area;
}

float HyperBox::maxDim() const {
  float maxSize = 0;
  for (size_t i = 0; i < dim; ++i) {
    float_t size = (it[i].sup - it[i].inf);
    maxSize = max(maxSize, size);
  }
  return maxSize;
}

bool HyperBox::anyDimBelow(const float delta) const {
  bool ret = false;
  for (size_t i = 0; i < dim; ++i) {
    float_t size = (it[i].sup - it[i].inf);
    ret = ret || (size <= delta);
  }
  return ret;
}


std::vector<HyperBox> HyperBox::split(int k) const {
  std::vector<std::vector<float_t>> splitPoints(dim);
  for (size_t i = 0; i < dim; ++i) {
    float_t delta = (it[i].sup - it[i].inf) / k;
    for (int j = 1; j <= k - 1; ++j) {
      splitPoints[i].push_back(it[i].inf + j * delta);
    }
  }

  std::vector<std::vector<Interval>> chunks(dim);
  for (size_t i = 0; i < dim; ++i) {
    float_t prev = it[i].inf;
    for (float_t x : splitPoints[i]) {
      chunks[i].emplace_back(prev, x);
      prev = x;
    }
    chunks[i].emplace_back(prev, it[i].sup);
  }

  std::vector<HyperBox> ret;
  for (size_t i = 0; i < dim; ++i) {
    std::vector<HyperBox> tmp = ret;
    ret.clear();

    for (const Interval& chunk : chunks[i]) {
      if (i == 0) {
        ret.push_back(HyperBox({chunk}));
      } else {
        for (HyperBox hbox : tmp) {
          HyperBox newBox = hbox;
          ++newBox.dim;
          newBox.it.push_back(chunk);
          ret.push_back(newBox);
        }
      }
    }
  }
  return ret;
}

std::ostream& operator << (std::ostream& os, const PointD& pt) {
  os << "(";
  for (size_t i = 0; i < pt.x.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << pt.x[i];
  }
  os << ")";
  return os;
}

std::ostream& operator << (std::ostream& os, const HyperBox& box) {
  os << "(";
  for (size_t i = 0; i < box.it.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << box.it[i];
  }
  os << ")";
  return os;
}
