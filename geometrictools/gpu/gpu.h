#include "gpu/image_gpu.h"
#include "gpu/interval.h"

#pragma once

void initGPU();
void cuda_err_print(cudaError_t error);
void cuda_err_check();
template <unsigned int blockSize> __device__ void warpReduce(volatile float *sdata, unsigned int tid);
__global__ void reduce(float *g_idata, float *g_odata, size_t n, size_t nSoft, size_t workPerThread);
template <unsigned int blockSize> __global__ void reduceInterval(IntervalGPU *g_idata, float *g_odata, size_t n, size_t nSoft);

template <unsigned int blockSize> __global__ void reduceIntervalDiff(IntervalGPU *g_idata_a, IntervalGPU *g_idata_b, float *g_odata, size_t n, size_t nSoft);
__device__ IntervalGPU d_make_IntervalGPU(float inf, float sup);
__device__ IntervalGPU d_make_IntervalGPU(const IntervalGPU& a);
__device__ IntervalGPU d_make_empty_IntervalGPU();
__device__ bool intervalGPU_is_empty(const IntervalGPU& a);
__device__ IntervalGPU abs(const IntervalGPU& it);
__device__ bool contains(const IntervalGPU& a, float x);
__device__ IntervalGPU operator - (const IntervalGPU &a);
__device__ IntervalGPU operator + (const IntervalGPU &a, const IntervalGPU &b);
__device__ IntervalGPU operator + (const float& a, const IntervalGPU &b);
__device__ IntervalGPU operator + (const IntervalGPU &a, const float& b);
__device__ IntervalGPU operator - (const float& a, const IntervalGPU &b);
__device__ IntervalGPU operator - (const IntervalGPU &a, const float& b);
__device__ IntervalGPU operator - (const IntervalGPU &a, const IntervalGPU &b);
__device__ IntervalGPU operator * (const float& a, const IntervalGPU &b);
__device__ IntervalGPU operator * (const IntervalGPU &a, const float& b);
__device__ IntervalGPU operator / (const IntervalGPU &a, const float& b);
__device__ IntervalGPU operator * (const IntervalGPU &a, const IntervalGPU &b);
__device__ IntervalGPU& operator += (IntervalGPU &a, const IntervalGPU &b);
__device__ IntervalGPU meet(const IntervalGPU& a, const IntervalGPU& b);

__device__ IntervalGPU join(const IntervalGPU& a, const IntervalGPU& b);

__device__ IntervalGPU BilinearInterpolation(const IntervalGPU& x,
                                             const IntervalGPU& y,
                                             const size_t ch,
                                             const IntervalGPU *data,
                                             dim3 dataDim);

__device__ int4 calculateBoundingBox(const IntervalGPU& x, const IntervalGPU& y,
                                     const size_t parity_x, const size_t parity_y);
__device__ size_t grid_rcc_to_idx(dim3 size, size_t r, size_t c, size_t ch);
__device__ int2 coord_to_rc(dim3 size, float x, float y);

__device__ IntervalGPU valueAt(const IntervalGPU* data, dim3 size, int x, int y, size_t ch, IntervalGPU default_value);

__device__ IntervalGPU valueAt(const IntervalGPU* data, dim3 size, int x, int y, size_t ch);

__device__ float2 getCoordinate(dim3 size, float r, float c);

__global__ void _resizeCropGPU(IntervalGPU* in, IntervalGPU* out, dim3 inDim, dim3 outDim,
                               dim3 preCropDim, size_t dx, size_t dy, const bool roundToInt);
void resizeCropGPU(const ImageImplGPU& in, ImageImplGPU& out, const bool roundToInt,
                   const size_t new_nRows, const size_t new_nCols,
                   const size_t dx, const size_t dy);
__global__ void _set(IntervalGPU* data, const IntervalGPU value, const dim3 size);

void setGPU(ImageImplGPU& image, const IntervalGPU value);
__global__ void _add(IntervalGPU* out, const IntervalGPU* lhs, const IntervalGPU* rhs, const dim3 size);
__global__ void _sub(IntervalGPU* out, const IntervalGPU* lhs, const IntervalGPU* rhs, const dim3 size);

void addGPU(ImageImplGPU& out, const ImageImplGPU& lhs, const ImageImplGPU& rhs);

void subGPU(ImageImplGPU& out, const ImageImplGPU& lhs, const ImageImplGPU& rhs);

vector<float> l2normGPU(const ImageImplGPU& image, size_t c);

vector<float> l2diffGPU(const ImageImplGPU& lhs, const ImageImplGPU& rhs, size_t c);


__device__ IntervalGPU R(const IntervalGPU i);

__global__ void _roundToInt(IntervalGPU* img, const dim3 size);

void roundToIntGPU(ImageImplGPU& image);

__global__ void _clip(IntervalGPU* img, const dim3 size, const float min_val, const float max_val);
void clipGPU(ImageImplGPU& image, const float min_val, const float max_val);

__device__ float4 bilinear_coefficients(int x_iter, int y_iter, float x, float y);

__global__ void _combineEmpty(const bool* d_isEmpty, bool* d_isEmptyCombined, const size_t s_isEmpty);

__global__ void _rotate(const IntervalGPU* in, IntervalGPU* out, const IntervalGPU* cgammap, const IntervalGPU* sgammap, const dim3 size, const bool roundToInt);

__global__ void _rotateM(const IntervalGPU* in, IntervalGPU* out, const IntervalGPU* cgammap, const IntervalGPU* sgammap, const dim3 size, const size_t filters, const bool roundToInt);

void rotateGPU(const ImageImplGPU& in, ImageImplGPU& out,
               const IntervalGPU* cgammap, const IntervalGPU* sgammap,
               const bool singleToMany, const bool roundToInt);

__global__ void _translate(const IntervalGPU* in, IntervalGPU* out, const IntervalGPU* dx, const IntervalGPU* dy, const dim3 size, const size_t params, const bool roundToInt);

void translateGPU(const ImageImplGPU& in, ImageImplGPU& out,
                  const IntervalGPU* dx, const IntervalGPU* dy,
                  const bool roundToInt);

__global__ void _center_crop(const IntervalGPU* in, IntervalGPU* out, const dim3 inSize, const dim3 outSize);

void center_cropGPU(const ImageImplGPU& in, const ImageImplGPU& out);


__global__ void _filter_vignetting(const IntervalGPU* in, IntervalGPU* out, const float* filter,
                                   const dim3 size, const size_t c, const float radiusSq);


void filter_vignettingGPU(const ImageImplGPU& in, const ImageImplGPU& out, const float* filter,
                          const size_t c, const float radiusSq);

__global__ void _vignetting(const IntervalGPU* in, IntervalGPU* out,
                            const dim3 size, const float radiusSq);


void vignettingGPU(const ImageImplGPU& in, const ImageImplGPU& out, const float radiusSq);

__global__ void _filterVignettingL2diffCGPU(float *odata,
                                            const IntervalGPU* lhs,
                                            const IntervalGPU* rhs,
                                            const ImageImplGPU* vingette,
                                            const dim3 size,
                                            const float radiusSq,
                                            const float* filter,
                                            const size_t c, /*  filter c */
                                            const size_t nrNorms,
                                            const size_t blocksPerNorm);



vector<float> filterVignettingL2diffCGPU(const ImageImplGPU& lhs, const ImageImplGPU& rhs,
                                         const ImageImplGPU* vingette,
                                         const float* filter,
                                         const size_t filter_c,
                                         const float radiusSq,
                                         const size_t c);




__global__ void _rect_vignetting(IntervalGPU* img, const dim3 size, const size_t filter_size);

void rect_vignettingGPU(ImageImplGPU& img, const size_t filter_size);



bool inverseGPU(const ImageImplGPU& in,
                ImageImplGPU** out,
                ImageImplGPU** vingette,
                Transform transform,
                const IntervalGPU transform_param0, // cgamma for rot, dx for trans
                const IntervalGPU transform_param1, // sgamma for rot, dy for trans
                const bool addIntegerError,
                const bool toIntegerValues,
                const size_t nrRefinements,
                const bool stopEarly,
                const bool computeVingette);

template <size_t C> // number of channels; this is a template to allow array allocation
    __global__ void _invert_pixel_on_bounding_box(const IntervalGPU* in,
                                                  IntervalGPU* out,
                                                  bool* isEmptyMask,
                                                  const dim3 size,
                                                  const bool refine,
                                                  const IntervalGPU* constraints,
                                                  const Transform transform,
                                                  const IntervalGPU transform_param0, //cgamma for rot, dx for trans
                                                  const IntervalGPU transform_param1, //sgamma for rot, dy for trans
                                                  const bool addIntegerError,
                                                  const bool toIntegerValues,
                                                  const bool computeVingette,
                                                  IntervalGPU* vingette);

  template <size_t C> // number of channels; this is a template to allow array allocation
  __device__ void _invert_pixel(const Corner center,
                                const dim3& size,
                                const float2& coord,
                                const IntervalGPU& coord_iterTx,
                                const IntervalGPU& coord_iterTy,
                                IntervalGPU* out,
                                const IntervalGPU* pixel_values,
                                const IntervalGPU* constraints,
                                const bool refine,
                                const bool debug);


__global__ void _customVingette(IntervalGPU* out, const IntervalGPU* in, const IntervalGPU* vingette, const dim3 size);
void customVingetteGPU(ImageImplGPU& out, const ImageImplGPU& in, const ImageImplGPU& vingette);


__global__ void _widthVingette(IntervalGPU* out, const IntervalGPU* in, const IntervalGPU* vingette, const dim3 size, const float treshold);
void widthVingetteGPU(ImageImplGPU& out, const ImageImplGPU& in, const ImageImplGPU& vingette, const float treshold);




__global__ void _threshold(IntervalGPU* out, const IntervalGPU* in, const dim3 size, const float val);
void thresholdGPU(ImageImplGPU& out, const ImageImplGPU& in, const float val);
__global__ void _pixelOr(IntervalGPU* out, const IntervalGPU* lhs, const IntervalGPU* rhs, const dim3 size);
void pixelOrGPU(ImageImplGPU& out, const ImageImplGPU& lhs, const ImageImplGPU& rhs);

__global__ void _pixelAnd(IntervalGPU* out, const IntervalGPU* lhs, const IntervalGPU* rhs, const dim3 size);
void pixelAndGPU(ImageImplGPU& out, const ImageImplGPU& lhs, const ImageImplGPU& rhs);



__global__ void _erode(IntervalGPU* out, const IntervalGPU* in, const dim3 size);
void erodeGPU(ImageImplGPU& out, const ImageImplGPU& in);
