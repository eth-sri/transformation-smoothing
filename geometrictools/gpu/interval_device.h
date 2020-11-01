#include "gpu/interval.h"

#pragma once
__device__ IntervalGPU d_make_IntervalGPU(float inf, float sup);
__device__ IntervalGPU d_make_IntervalGPU(const IntervalGPU& a);
__device__ IntervalGPU d_make_empty_IntervalGPU();
__device__ bool intervalGPU_is_empty(const IntervalGPU& a);
__device__ bool contains(const IntervalGPU& a, float x);

__device__ IntervalGPU meet(const IntervalGPU& a, const IntervalGPU& b);
__device__ IntervalGPU join(const IntervalGPU& a, const IntervalGPU& b);

__device__ IntervalGPU operator - (const IntervalGPU &a);
__device__ IntervalGPU operator + (const IntervalGPU &a, const IntervalGPU &b);
__device__ IntervalGPU operator - (const IntervalGPU &a, const IntervalGPU &b);
__device__ IntervalGPU operator + (const float& a, const IntervalGPU &b);
__device__ IntervalGPU operator + (const IntervalGPU &a, const float& b);
__device__ IntervalGPU operator - (const float& a, const IntervalGPU &b);
__device__ IntervalGPU operator - (const IntervalGPU &a, const float& b);
__device__ IntervalGPU operator * (const float& a, const IntervalGPU &b);
__device__ IntervalGPU operator * (const IntervalGPU &a, const float& b);
__device__ IntervalGPU operator * (const IntervalGPU &a, const IntervalGPU &b);

__device__ IntervalGPU& operator += (IntervalGPU &a, const IntervalGPU &b);

__device__ IntervalGPU abs(const IntervalGPU& it);
// IntervalGPU square(const IntervalGPU& it);
