#include <cuda.h>
#pragma once

struct alignas(8) IntervalGPU {
  float inf;
  float sup;
};
