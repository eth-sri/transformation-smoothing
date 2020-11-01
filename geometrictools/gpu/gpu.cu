#include "gpu/gpu.h"
#include "gpu/interval_device.h"
#include <math_constants.h>
#include <cmath>
#include <chrono>
#include <stdio.h>
#include <omp.h>

__constant__ float d_pixelVals[256];
bool isInit = false;

void initGPU(){
  int numGPUs;
  cudaGetDeviceCount(&numGPUs);
  for(int gpu = 0; gpu < numGPUs; ++gpu) {
    cudaSetDevice(gpu);
    cudaMemcpyToSymbol(d_pixelVals, pixelVals, 256*sizeof(float));
  }
  isInit = true;  
};

void cuda_err_print(cudaError_t error) {
  if(error != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}  

void cuda_err_check() {
  cudaError_t error = cudaGetLastError();
  cuda_err_print(error);
}  

// efficient reduction
// code based on http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

template <unsigned int blockSize> __device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
  if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
  if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
  if (blockSize >=   8) sdata[tid] += sdata[tid +  4];
  if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
  if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
}


__global__ void reduce(float *g_idata, float *g_odata, size_t n, size_t nSoft, size_t workPerThread) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  sdata[tid] = 0;


  
  for(size_t k = 0; k < workPerThread; ++k) {
    const size_t idx = workPerThread*tid + k;
    if (idx < nSoft) {
      sdata[tid] += g_idata[bid * nSoft + idx];
    }
  }
  __syncthreads();
  if (n >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
  if (n >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (n >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (n >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
  if (tid < 32) {
       if (n >= 1024) warpReduce<1024>(sdata, tid);
       else if (n >= 512) warpReduce<512>(sdata, tid);
       else if (n >= 256) warpReduce<256>(sdata, tid);
       else if (n >= 128) warpReduce<128>(sdata, tid);
       else if (n >= 64) warpReduce<64>(sdata, tid);
       else if (n >= 32) warpReduce<32>(sdata, tid);
       else if (n >= 16) warpReduce<16>(sdata, tid);
       else if (n < 16) {
         if (tid == 0) {
           for(size_t k = 1; k < n; ++k) {
             sdata[0] += sdata[k];
           }
         }
       }
  }
  if (tid == 0) g_odata[bid] = sqrt(sdata[0]);
}

template <unsigned int blockSize> __global__ void reduceInterval(IntervalGPU *g_idata, float *g_odata, size_t n, size_t nSoft) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) {
    float a = 0;
    float b = 0;
      
    if (i < nSoft) {
      a = max(g_idata[i].inf * g_idata[i].inf, g_idata[i].sup * g_idata[i].sup);
    }
    if (i + blockSize < nSoft) {
      b = max(g_idata[i+blockSize].inf * g_idata[i+blockSize].inf, g_idata[i+blockSize].sup * g_idata[i+blockSize].sup);
    }
    sdata[tid] += a + b;
    i += gridSize;
  }
  __syncthreads();
  if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
  if (tid < 32) {
    warpReduce<blockSize>(sdata, tid);
  }
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize> __global__ void reduceIntervalDiff(IntervalGPU *g_idata_a, IntervalGPU *g_idata_b, float *g_odata, size_t n, size_t nSoft) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) {
    float a = 0;
    float b = 0;
      
    if (i < nSoft) {
      IntervalGPU tmp = g_idata_a[i] - g_idata_b[i];
      a = max(tmp.inf * tmp.inf, tmp.sup * tmp.sup);
    }
    if (i + blockSize < nSoft) {
      IntervalGPU tmp = g_idata_a[i+blockSize] - g_idata_b[i+blockSize];
      b = max(tmp.inf * tmp.inf, tmp.sup * tmp.sup);
    }
    sdata[tid] += a + b;
    i += gridSize;
  }
  __syncthreads();
  if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
  if (tid < 32) {
    warpReduce<blockSize>(sdata, tid);
  }
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__device__ IntervalGPU d_make_IntervalGPU(float inf, float sup) {
  if (!(inf <= sup + Constants::EPS)) printf("%.20f %.20f\n", inf, sup); assert(inf <= sup + Constants::EPS);
  return {inf, sup};
}

__device__ IntervalGPU d_make_IntervalGPU(const IntervalGPU& a) {
  return {a.inf, a.sup};
}

__device__ IntervalGPU d_make_empty_IntervalGPU() {
  return {CUDART_INF_F, -CUDART_INF_F};
}

__device__ bool intervalGPU_is_empty(const IntervalGPU& a){
  return a.inf == CUDART_INF_F &&
    a.sup == -CUDART_INF_F;
}

__device__ IntervalGPU abs(const IntervalGPU& it) {
  if (it.sup < 0) {
    return d_make_IntervalGPU(-it.sup, -it.inf);
  } else if (it.inf > 0) {
    return d_make_IntervalGPU(it.inf, it.sup);
  }
  return d_make_IntervalGPU(0, max(-it.inf, it.sup));
}

__device__ bool contains(const IntervalGPU& a, float x) {
  return x >= a.inf && x <= a.sup;
}

__device__ IntervalGPU operator - (const IntervalGPU &a) {
  return d_make_IntervalGPU(-a.sup, -a.inf);
}

__device__ IntervalGPU operator + (const IntervalGPU &a, const IntervalGPU &b) {
  IntervalGPU r = d_make_IntervalGPU(__fadd_rd (a.inf, b.inf), __fadd_ru (a.sup, b.sup));
  if (!(r.inf <= r.sup)) printf("[%.20f %.20f] +  [%.20f %.20f] =  [%.20f %.20f]\n", a.inf, a.sup, b.inf, b.sup, r.inf, r.sup); assert(r.inf <= r.sup + Constants::EPS);
  return r;
}

__device__ IntervalGPU operator + (const float& a, const IntervalGPU &b) {
  IntervalGPU r = d_make_IntervalGPU(a, a);
  r = r + b;
  if (!(r.inf <= r.sup)) printf("%.20f %.20f\n", r.inf, r.sup);assert(r.inf <= r.sup);
  return r;
}

__device__ IntervalGPU operator + (const IntervalGPU &a, const float& b) {
  return b + a;
}

__device__ IntervalGPU operator - (const float& a, const IntervalGPU &b) {
  return a + (-b);
}

__device__ IntervalGPU operator - (const IntervalGPU &a, const float& b) {
  return a + (-b);
}

__device__ IntervalGPU operator - (const IntervalGPU &a, const IntervalGPU &b) {
  return a + (-b);
}

__device__ IntervalGPU operator * (const float& a, const IntervalGPU &b) {
  if (a > 0) {
    IntervalGPU ret = d_make_IntervalGPU(__fmul_rd (a, b.inf), __fmul_ru(a, b.sup)); 
    if (!(ret.inf <= ret.sup)) printf("%.20f * [%.20f %.20f] = [%.20f %.20f]\n", a, b.inf, b.sup, ret.inf, ret.sup);  assert(ret.inf <= ret.sup);
    return ret;
  }
  
  IntervalGPU ret = d_make_IntervalGPU(__fmul_rd (a, b.sup), __fmul_ru (a, b.inf));
  if (!(ret.inf <= ret.sup)) printf("%.20f * [%.20f %.20f] = [%.20f %.20f]\n", a, b.inf, b.sup, ret.inf, ret.sup);  assert(ret.inf <= ret.sup);
  return d_make_IntervalGPU(__fmul_rd (b.sup, a), __fmul_ru (b.inf, a));
}

__device__ IntervalGPU operator * (const IntervalGPU &a, const float& b) {
  return b * a;
}

__device__ IntervalGPU operator / (const IntervalGPU &a, const float& b) {
  assert(b != 0);
  return (1/b) * a;
}

__device__ IntervalGPU operator * (const IntervalGPU &a, const IntervalGPU &b) {
  IntervalGPU tmp1 = a * b.inf;
  IntervalGPU tmp2 = a * b.sup;
  return d_make_IntervalGPU(min(tmp1.inf, tmp2.inf), max(tmp1.sup, tmp2.sup));
}

__device__ IntervalGPU& operator += (IntervalGPU &a, const IntervalGPU &b) {
  a.inf = __fadd_rd (a.inf, b.inf);
  a.sup = __fadd_rd (a.sup, b.sup);
  return a;
}

__device__ IntervalGPU meet(const IntervalGPU& a, const IntervalGPU& b) {
  if (intervalGPU_is_empty(a) || intervalGPU_is_empty(b)) {
    return d_make_empty_IntervalGPU();
  }
  if (a.inf > b.sup || b.inf > a.sup) {
    return d_make_empty_IntervalGPU();
  }
  return d_make_IntervalGPU(max(a.inf, b.inf), min(a.sup, b.sup));
}

__device__ IntervalGPU join(const IntervalGPU& a, const IntervalGPU& b) {
  const bool aE = intervalGPU_is_empty(a);
  const bool bE = intervalGPU_is_empty(b);

  if (aE && bE) return d_make_empty_IntervalGPU();
  else if (aE) return d_make_IntervalGPU(b.inf, b.sup);
  else if (bE) return d_make_IntervalGPU(a.inf, a.sup);
  
  return d_make_IntervalGPU(min(a.inf, b.inf), max(a.sup, b.sup));
}

__device__ IntervalGPU BilinearInterpolation(const IntervalGPU& x,
                                             const IntervalGPU& y,
                                             const size_t ch,
                                             const IntervalGPU *data,
                                             dim3 dataDim) {
  assert(data != 0);
  IntervalGPU ret = d_make_empty_IntervalGPU();
  int parity_x = (dataDim.y - 1) % 2;
  int parity_y = (dataDim.x - 1) % 2;
  int lo_x, hi_x, lo_y, hi_y;

  //computes bounding box of possible concrete coordinates
  int4 bounds = calculateBoundingBox(x, y, parity_x, parity_y);
  lo_x = bounds.x;
  hi_x = bounds.y;
  lo_y = bounds.z;
  hi_y = bounds.w;
    
  IntervalGPU values[50];
  size_t cc = 0;
  
  //traverse all interpolation regions
  size_t cnt = 0;
  for (int x1 = lo_x; x1 <= hi_x; x1 += 2) {
    for (int y1 = lo_y; y1 <= hi_y; y1 += 2) {
      
      //intersect coordinates with interpolation region
      IntervalGPU x_box = meet(x, d_make_IntervalGPU(x1, x1 + 2));       
      IntervalGPU y_box = meet(y, d_make_IntervalGPU(y1, y1 + 2));      
      if (intervalGPU_is_empty(x_box) || intervalGPU_is_empty(y_box)) {
        cnt++;
        continue;
      }
      assert(x.inf <= x_box.inf);
      assert(x_box.sup <= x.sup);
      assert(y.inf <= y_box.inf);
      assert(y_box.sup <= y.sup);
      
      //compute coordinates of 4 corners of interpolation region
      float alpha, beta, gamma, delta;
      IntervalGPU a = valueAt(data, dataDim, x1, y1, ch);
      IntervalGPU b = valueAt(data, dataDim, x1, y1 + 2, ch);
      IntervalGPU c = valueAt(data, dataDim, x1 + 2, y1, ch);
      IntervalGPU d = valueAt(data, dataDim, x1 + 2, y1 + 2, ch);
      alpha = a.inf;
      beta = b.inf;
      gamma = c.inf;
      delta = d.inf;
      
      //use formula for bilinear interpolation to obtain all possible coordinates      
      IntervalGPU tmp_low =  0.25f * (alpha * (x1 + 2 - x_box) * (y1 + 2 - y_box)
                                      + beta * (x1 + 2 - x_box) * (y_box - y1)
                                      + gamma * (x_box - x1) * (y1 + 2 - y_box)
                                      + delta * (x_box - x1) * (y_box - y1));
      ret = join(ret, tmp_low);
     
      alpha = a.sup;
      beta = b.sup;
      gamma = c.sup;
      delta = d.sup;
      IntervalGPU tmp_high = 0.25f * (alpha * (x1 + 2 - x_box) * (y1 + 2 - y_box)
                                      + beta * (x1 + 2 - x_box) * (y_box - y1)
                                      + gamma * (x_box - x1) * (y1 + 2 - y_box)
                                      + delta * (x_box - x1) * (y_box - y1));

      ret = join(ret, tmp_high);
    }
  }
  assert(!intervalGPU_is_empty(ret));  
  return ret;
}


__device__ int4 calculateBoundingBox(const IntervalGPU& x, const IntervalGPU& y,
                                     const size_t parity_x, const size_t parity_y) {
  int lo_x = (int)floorf(x.inf - 2*Constants::EPS);
  int hi_x = (int)ceilf(x.sup + 2*Constants::EPS);
  int lo_y = (int)floorf(y.inf - 2*Constants::EPS);
  int hi_y = (int)ceilf(y.sup + 2*Constants::EPS);

  if (abs(lo_x) % 2 != parity_x) {
    --lo_x;
  }
  if (abs(hi_x) % 2 != parity_x) {
    ++hi_x;
  }
  if (abs(lo_y) % 2 != parity_y) {
    --lo_y;
  }
  if (abs(hi_y) % 2 != parity_y) {
    ++hi_y;
  }

  return make_int4(lo_x, hi_x, lo_y, hi_y);
}

__device__ size_t grid_rcc_to_idx(dim3 size, size_t r, size_t c, size_t ch) { 
  assert(size.x > 0);
  assert(size.y > 0);
  assert(size.z > 0);
  const size_t out = ch * size.x * size.y + r * size.y + c;
  return out;
}

__device__ int2 coord_to_rc(dim3 size, float x, float y) { 
  assert(size.x > 0);
  assert(size.y > 0);
  assert(size.z > 0);
  int c = int(int(x + (size.y - 1)) / 2);
  int r = int(int(y + (size.x - 1)) / 2);
  int2 out = make_int2(r, c);
  return out;
}

__device__ IntervalGPU valueAt(const IntervalGPU* data, dim3 size, int x, int y, size_t ch, IntervalGPU default_value) {
  assert(unsigned(x % 2) != size.y % 2 && unsigned(y % 2) != size.x % 2 && "Off grid coordinates"); 
  const int c = (x + (size.y - 1)) / 2;
  const int r = (y + (size.x - 1)) / 2;
  if (r < 0 || r >= signed(size.x) || c < 0 || c >= signed(size.y)) {
    return default_value;
  }
  size_t m = grid_rcc_to_idx(size, (size_t) r, (size_t) c, ch);
  return data[m];
}


__device__ IntervalGPU valueAt(const IntervalGPU* data, dim3 size, int x, int y, size_t ch) {
  return valueAt(data, size, x, y, ch, d_make_IntervalGPU(0, 0));
}

__device__ float2 getCoordinate(dim3 size, float r, float c){
  return make_float2(2 * c - (size.y - 1), 2 * r - (size.x - 1));
}


__global__ void _resizeCropGPU(IntervalGPU* in, IntervalGPU* out, dim3 inDim, dim3 outDim,
                               dim3 preCropDim, size_t dx, size_t dy, const bool roundToInt) {
  size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;  
  
  if (row < outDim.x && col < outDim.y && chan < outDim.z) {
    const size_t m = grid_rcc_to_idx(outDim, row, col, chan);
    const size_t r = row + dx;
    const size_t c = col + dy;
    float2 coord = getCoordinate(inDim,
                                 r * inDim.x / (float) preCropDim.x,
                                 c * inDim.y / (float) preCropDim.y);
    IntervalGPU x = d_make_IntervalGPU(coord.x, coord.x);
    IntervalGPU y = d_make_IntervalGPU(coord.y, coord.y);
    out[m] = BilinearInterpolation(x, y, chan, in, inDim);
    if (roundToInt) { 
      out[m] = R(out[m]);
    }
  }
}

void resizeCropGPU(const ImageImplGPU& in, ImageImplGPU& out, const bool roundToInt,
                   const size_t new_nRows, const size_t new_nCols,
                   const size_t dx, const size_t dy) {
  assert(in.device == out.device);
  int dev;
  cudaGetDevice(&dev);
  assert(in.device == dev);

  assert(in.nCols >= out.nCols);
  assert(in.nRows >= out.nRows); 
  assert(in.nChannels == out.nChannels);
  assert(out.data != 0);
  assert(in.data != 0);
  assert(out.nCols + 2 * dy == new_nCols || out.nCols + 2 * dy + 1 == new_nCols);
  assert(out.nRows + 2 * dx == new_nRows || out.nRows + 2 * dx + 1 == new_nRows);

  
  const size_t block_z = out.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z)) / 2; //factor 2 is needed to have enough registers per thread
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) out.nRows / (float) block_xy), ceilf( (float) out.nCols / (float) block_xy), 1);
 
  dim3 inDim(in.nRows, in.nCols, in.nChannels);
  dim3 preCropDim(new_nRows, new_nCols, out.nChannels);
  dim3 outDim(out.nRows, out.nCols, out.nChannels);  
  _resizeCropGPU<<<g, b>>>(in.data, out.data, inDim, outDim, preCropDim, dx, dy, roundToInt);
  cuda_err_check();
}

__global__ void _set(IntervalGPU* data, const IntervalGPU value, const dim3 size) {
  size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;    
  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    data[m] = value;
  }
}


void setGPU(ImageImplGPU& image, const IntervalGPU value) {
  int dev;
  cudaGetDevice(&dev);
  assert(image.device == dev);
  const size_t block_z = image.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) image.nRows / (float) block_xy), ceilf( (float) image.nCols / (float) block_xy), 1);
  assert(image.data != 0); 
  dim3 size(image.nRows, image.nCols, image.nChannels);
  _set<<<g, b>>>(image.data, value, size);
  cuda_err_check();
}

__global__ void _add(IntervalGPU* out, const IntervalGPU* lhs, const IntervalGPU* rhs, const dim3 size) {
  size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;    
  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    out[m] = lhs[m] + rhs[m];
    assert(out[m].inf <= out[m].sup + Constants::EPS);
  }
}

__global__ void _sub(IntervalGPU* out, const IntervalGPU* lhs, const IntervalGPU* rhs, const dim3 size) {
  size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;    
  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    out[m] = lhs[m] - rhs[m];
    assert(out[m].inf <= out[m].sup + Constants::EPS);
  }
}


void addGPU(ImageImplGPU& out, const ImageImplGPU& lhs, const ImageImplGPU& rhs) {
  assert(lhs.device == rhs.device);
  assert(lhs.device == out.device);
  int dev;
  cudaGetDevice(&dev);
  assert(lhs.device == dev);
  const size_t block_z = out.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) out.nRows / (float) block_xy), ceilf( (float) out.nCols / (float) block_xy), 1);
  assert(out.data != 0);
  assert(lhs.data != 0);
  assert(rhs.data != 0);
  assert(out.nRows == rhs.nRows);
  assert(out.nCols == rhs.nCols);
  assert(out.nChannels == rhs.nChannels);
  assert(out.nRows == lhs.nRows);
  assert(out.nCols == lhs.nCols);
  assert(out.nChannels == lhs.nChannels);
  dim3 size(out.nRows, out.nCols, out.nChannels);
  _add<<<g, b>>>(out.data, lhs.data, rhs.data, size);
  cuda_err_check();
}

void subGPU(ImageImplGPU& out, const ImageImplGPU& lhs, const ImageImplGPU& rhs) {
  assert(lhs.device == rhs.device);
  assert(lhs.device == out.device);
  int dev;
  cudaGetDevice(&dev);
  assert(lhs.device == dev);  
  const size_t block_z = out.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) out.nRows / (float) block_xy), ceilf( (float) out.nCols / (float) block_xy), 1);
  assert(out.data != 0);
  assert(lhs.data != 0);
  assert(rhs.data != 0);
  assert(out.nRows == rhs.nRows);
  assert(out.nCols == rhs.nCols);
  assert(out.nChannels == rhs.nChannels);
  assert(out.nRows == lhs.nRows);
  assert(out.nCols == lhs.nCols);
  assert(out.nChannels == lhs.nChannels);
  dim3 size(out.nRows, out.nCols, out.nChannels);
  _sub<<<g, b>>>(out.data, lhs.data, rhs.data, size);
  cuda_err_check();
}

vector<float> l2normGPU(const ImageImplGPU& image, size_t c) {
  assert(image.data != 0);
  assert(c <= image.nChannels);
  assert(image.nChannels % c == 0);
  const size_t nrNorms = image.nChannels / c;
  const size_t s = image.nRows * image.nCols * c;
  const size_t n = (size_t) pow(2, ceilf(log2(s))); //find next largest power of two
  assert(n >= 128); //if problem is smaller do it differently; GPU won't help
  assert(n >= s);

  float *gpu_result[nrNorms];
  cudaStream_t streams[nrNorms];
  
  size_t gridDim = n;  
  for(size_t i = 0; i < nrNorms; ++i) {
    cudaStreamCreate(&streams[i]);
    IntervalGPU *start = image.data + i * s;
    if (n >= 2048) {
      gridDim = n / (1024*2);
      cuda_err_print(cudaMalloc((void**)&gpu_result[i], gridDim*sizeof(float)));  
      reduceInterval<1024><<<gridDim, 1024, 1024*2*sizeof(float), streams[i]>>>(start, gpu_result[i], n, s);
    }  else if (n >= 1024) {
      gridDim = n / (512*2);
      cuda_err_print(cudaMalloc((void**)&gpu_result[i], gridDim*sizeof(float)));  
      reduceInterval<512><<<gridDim, 512, 512*2*sizeof(float), streams[i]>>>(start, gpu_result[i], n, s);
    }  else if (n >= 512) {
      gridDim = n / (256*2);
      cuda_err_print(cudaMalloc((void**)&gpu_result[i], gridDim*sizeof(float)));  
      reduceInterval<256><<<gridDim, 256, 256*2*sizeof(float), streams[i]>>>(start, gpu_result[i], n, s);
    }  else if (n >= 256) {
      gridDim = n / (128*2);
      cuda_err_print(cudaMalloc((void**)&gpu_result[i], gridDim*sizeof(float)));  
      reduceInterval<128><<<gridDim, 128, 128*2*sizeof(float), streams[i]>>>(start, gpu_result[i], n, s);
    } else if (n >= 128) {
      gridDim = n / (64*2);
      cuda_err_print(cudaMalloc((void**)&gpu_result[i], gridDim*sizeof(float)));  
      reduceInterval<64><<<gridDim, 64, 64*2*sizeof(float), streams[i]>>>(start, gpu_result[i], n, s);
    } else {
      // TODO sequential solution of n < 128
      assert(false);
    }
  }

  for(size_t i = 0; i < nrNorms; ++i) {
    cuda_err_print(cudaStreamSynchronize(streams[i]));
    cuda_err_print(cudaStreamDestroy(streams[i]));
  }
  cuda_err_check();

  vector<float> norms;
  for(size_t i = 0; i < nrNorms; ++i) {
    assert(gpu_result[i] != 0);
    float result[gridDim];
    cudaStreamSynchronize(0);
    cuda_err_print(cudaMemcpy(result, gpu_result[i], gridDim*sizeof(float), cudaMemcpyDeviceToHost));
    cuda_err_print(cudaFree(gpu_result[i]));
    
    float out = 0;
    for(size_t k = 0; k < gridDim; ++k) {
      out += result[k];
    }
  
    norms.push_back(sqrt(out));
  }

  return norms;  
}


vector<float> l2diffGPU(const ImageImplGPU& lhs, const ImageImplGPU& rhs, size_t c) {
  assert(lhs.data != 0);
  assert(rhs.data != 0);
  assert(lhs.nRows == rhs.nRows);
  assert(lhs.nCols == rhs.nCols);
  assert(lhs.nChannels == rhs.nChannels);
  assert(c <= lhs.nChannels);
  assert(lhs.nChannels % c == 0);
  const size_t nrNorms = lhs.nChannels / c;
  const size_t s = lhs.nRows * lhs.nCols * c;
  const size_t n = (size_t) pow(2, ceilf(log2(s))); //find next largest power of two
  assert(n >= 128); //if problem is smaller do it differently; GPU won't help
  assert(n >= s);

  float *gpu_result[nrNorms];
  cudaStream_t streams[nrNorms];
  
  size_t gridDim = n;  
  for(size_t i = 0; i < nrNorms; ++i) {
    cudaStreamCreate(&streams[i]);
    IntervalGPU *startLHS = lhs.data + i * s;
    IntervalGPU *startRHS = rhs.data + i * s;
    if (n >= 2048) {
      gridDim = n / (1024*2);
      cuda_err_print(cudaMalloc((void**)&gpu_result[i], gridDim*sizeof(float)));  
      reduceIntervalDiff<1024><<<gridDim, 1024, 1024*2*sizeof(float), streams[i]>>>(startLHS, startRHS, gpu_result[i], n, s);
    }  else if (n >= 1024) {
      gridDim = n / (512*2);
      cuda_err_print(cudaMalloc((void**)&gpu_result[i], gridDim*sizeof(float)));  
      reduceIntervalDiff<512><<<gridDim, 512, 512*2*sizeof(float), streams[i]>>>(startLHS, startRHS, gpu_result[i], n, s);
    }  else if (n >= 512) {
      gridDim = n / (256*2);
      cuda_err_print(cudaMalloc((void**)&gpu_result[i], gridDim*sizeof(float)));  
      reduceIntervalDiff<256><<<gridDim, 256, 256*2*sizeof(float), streams[i]>>>(startLHS, startRHS, gpu_result[i], n, s);
    }  else if (n >= 256) {
      gridDim = n / (128*2);
      cuda_err_print(cudaMalloc((void**)&gpu_result[i], gridDim*sizeof(float)));  
      reduceIntervalDiff<128><<<gridDim, 128, 128*2*sizeof(float), streams[i]>>>(startLHS, startRHS, gpu_result[i], n, s);
    } else if (n >= 128) {
      gridDim = n / (64*2);
      cuda_err_print(cudaMalloc((void**)&gpu_result[i], gridDim*sizeof(float)));  
      reduceIntervalDiff<64><<<gridDim, 64, 64*2*sizeof(float), streams[i]>>>(startLHS, startRHS, gpu_result[i], n, s);
    } else {
      // TODO sequential solution of n < 128
      assert(false);
    }
  }

  for(size_t i = 0; i < nrNorms; ++i) {
    cuda_err_print(cudaStreamSynchronize(streams[i]));
    cuda_err_print(cudaStreamDestroy(streams[i]));
  }
  cuda_err_check();

  vector<float> norms(nrNorms);
  for(size_t i = 0; i < nrNorms; ++i) {
    assert(gpu_result[i] != 0);
    float result[gridDim];
    cudaStreamSynchronize(0);
    cuda_err_print(cudaMemcpy(result, gpu_result[i], gridDim*sizeof(float), cudaMemcpyDeviceToHost));
    cuda_err_print(cudaFree(gpu_result[i]));
 
    float out = 0;
    for(size_t k = 0; k < gridDim; ++k) {
      out += result[k];
    }
  
    norms[i] = sqrt(out);
  }

  return norms;  
}


__device__ IntervalGPU R(const IntervalGPU i) {
  IntervalGPU in = meet(i, d_make_IntervalGPU(0.0f, 1.0f));
  float inf = in.inf * 255.0f;  
  int inf_lower = floorf(inf);
  if (inf_lower == roundf(inf)) {
    inf = d_pixelVals[inf_lower];
  } else {
    inf = d_pixelVals[min(255, inf_lower + 1)];
  }

  float sup = in.sup * 255.0f;  
  int sup_lower = floorf(sup);
  if (sup_lower == roundf(sup)) {
    sup = d_pixelVals[sup_lower];
  } else {
    sup = d_pixelVals[min(255, sup_lower + 1)];
  }

  assert(inf <= sup);
  return d_make_IntervalGPU(inf, sup);  
}

__global__ void _roundToInt(IntervalGPU* img, const dim3 size) {
  size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;    
  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    if (!intervalGPU_is_empty(img[m])) {       
      img[m] = R(img[m]);
    }    
  }
}

void roundToIntGPU(ImageImplGPU& image) {
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(image.device == dev);
  const size_t block_z = image.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) image.nRows / (float) block_xy), ceilf( (float) image.nCols / (float) block_xy), 1);
  assert(image.data != 0);
  dim3 size(image.nRows, image.nCols, image.nChannels);
  _roundToInt<<<g, b>>>(image.data, size);
  cuda_err_check();
}

__global__ void _clip(IntervalGPU* img, const dim3 size, const float min_val, const float max_val) {
  size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;    
  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    if (!intervalGPU_is_empty(img[m])) {       
      img[m] = d_make_IntervalGPU(min(max(min_val, img[m].inf), max_val),
                                  min(max(min_val, img[m].sup), max_val));      
    }    
  }
}

void clipGPU(ImageImplGPU& image, const float min_val, const float max_val) {
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(image.device == dev);
  const size_t block_z = image.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) image.nRows / (float) block_xy), ceilf( (float) image.nCols / (float) block_xy), 1);
  assert(image.data != 0);
  dim3 size(image.nRows, image.nCols, image.nChannels);
  _clip<<<g, b>>>(image.data, size, min_val, max_val);
  cuda_err_check();
}


__device__ float4 bilinear_coefficients(int x_iter, int y_iter, float x, float y) {
  assert(x_iter <= x && x <= x_iter + 2 && y_iter <= y && y <= y_iter + 2);
  float a_alpha = (x_iter + 2 - x) * (y_iter + 2 - y) / 4;
  float a_beta = (x_iter + 2 - x) * (y - y_iter) / 4;
  float a_gamma = (x - x_iter) * (y_iter + 2 - y) / 4;
  float a_delta = (x - x_iter) * (y - y_iter) / 4;
  return make_float4(a_alpha, a_beta, a_gamma, a_delta);
}


__global__ void _combineEmpty(const bool* d_isEmpty, bool* d_isEmptyCombined, const size_t s_isEmpty) {
  extern __shared__ bool isEmpty[];
  const size_t tid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t blockSize = blockDim.x;

  const size_t idx = 16 * blockSize * bid + tid * 16;
  if (idx < s_isEmpty) {
    isEmpty[tid] = d_isEmpty[idx];
    for(size_t k = 1; k < 16; ++k) {
      if (idx + k < s_isEmpty) {
        isEmpty[tid] = isEmpty[tid] || d_isEmpty[idx + k];
      }
    }
  } else {
    isEmpty[tid] = false;
  }
  __syncthreads();
  if (blockSize >= 512) {if (tid < 512 && tid + 512 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 512]; } __syncthreads();}
  if (blockSize >= 256) {if (tid < 256 && tid + 256 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 256]; } __syncthreads();}
  if (blockSize >= 128) {if (tid < 128 && tid + 128 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 128]; } __syncthreads();}
  if (blockSize >= 64) {if (tid < 64 && tid + 64 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 64]; } __syncthreads();}
  if (blockSize >= 32) {if (tid < 32 && tid + 32 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 32]; } __syncthreads();}
  if (blockSize >= 16) {if (tid < 16 && tid + 16 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 16]; } __syncthreads();}

  if(tid == 0) {
    for(size_t k = 1; k < 16; ++k) {
      isEmpty[0] = isEmpty[0] || isEmpty[k]; 
    }
    d_isEmptyCombined[bid] = isEmpty[0];
  }
}


__global__ void _rotate(const IntervalGPU* in, IntervalGPU* out, const IntervalGPU* cgammap, const IntervalGPU* sgammap, const dim3 size, const bool roundToInt) {
  const size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  const size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  const size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;
  
  if (row < size.x && col < size.y && chan < size.z) {
    const IntervalGPU cgamma = cgammap[chan];
    const IntervalGPU sgamma = sgammap[chan];
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    const float2 coord = getCoordinate(size, row, col);
    const IntervalGPU x = coord.x * cgamma - coord.y * sgamma;
    const IntervalGPU y = coord.x * sgamma + coord.y * cgamma;
    out[m] = BilinearInterpolation(x, y, chan, in, size);
    if (roundToInt) { 
      out[m] = R(out[m]);
    }
  }
}

__global__ void _rotateM(const IntervalGPU* in, IntervalGPU* out, const IntervalGPU* cgammap, const IntervalGPU* sgammap, const dim3 size, const size_t filters, const bool roundToInt) {
  const size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  const size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  const size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;

  if (row < size.x && col < size.y && chan < size.z*filters) {
    const size_t source_chan = chan % size.z;
    const size_t transform_id = chan / size.z;
    assert(transform_id < filters);
    const IntervalGPU cgamma = cgammap[transform_id];
    const IntervalGPU sgamma = sgammap[transform_id];
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    const float2 coord = getCoordinate(size, row, col);
    const IntervalGPU x = coord.x * cgamma - coord.y * sgamma;
    const IntervalGPU y = coord.x * sgamma + coord.y * cgamma;
    out[m] = BilinearInterpolation(x, y, source_chan, in, size);
    if (roundToInt) { 
      out[m] = R(out[m]);
    }
  }
}


void rotateGPU(const ImageImplGPU& in, ImageImplGPU& out,
               const IntervalGPU* cgammap, const IntervalGPU* sgammap,
               const bool singleToMany, const bool roundToInt) {
  assert(in.device == out.device);
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(in.device == dev);
  const size_t block_z = out.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z)) / 2;
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) out.nRows / (float) block_xy), ceilf( (float) out.nCols / (float) block_xy), 1);
  assert(out.data != 0);
  assert(in.data != 0);
  assert(in.nCols == out.nCols);
  assert(in.nRows == out.nRows);

  dim3 size(in.nRows, in.nCols, in.nChannels);
  
  if (singleToMany) {
    assert(out.nChannels % in.nChannels == 0);
    _rotateM<<<g, b>>>(in.data, out.data, cgammap, sgammap, size, out.nChannels / in.nChannels, roundToInt);
  } else {
    assert(in.nChannels == out.nChannels);
    _rotate<<<g, b>>>(in.data, out.data, cgammap, sgammap, size, roundToInt);
  }
  
  cuda_err_check();

}

__global__ void _translate(const IntervalGPU* in, IntervalGPU* out, const IntervalGPU* dx, const IntervalGPU* dy, const dim3 size, const size_t params, const bool roundToInt) {
  const size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  const size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  const size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;

  
  if (row < size.x && col < size.y && chan < size.z*params) {
    const size_t source_chan = chan % size.z;
    const size_t transform_id = chan / size.z;
    assert(transform_id < params);
    const IntervalGPU cdx = dx[transform_id];
    const IntervalGPU cdy = dy[transform_id];
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    const float2 coord = getCoordinate(size, row, col);
    const IntervalGPU x = coord.x + 2*cdx;
    const IntervalGPU y = coord.y + 2*cdy;
    out[m] = BilinearInterpolation(x, y, source_chan, in, size);
    if (roundToInt) { 
      out[m] = R(out[m]);
    }
  }
}


void translateGPU(const ImageImplGPU& in, ImageImplGPU& out,
                  const IntervalGPU* dx, const IntervalGPU* dy,
                  const bool roundToInt) {
  assert(in.device == out.device);
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(in.device == dev);
  const size_t block_z = out.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z)) / 2;
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) out.nRows / (float) block_xy), ceilf( (float) out.nCols / (float) block_xy), 1);
  assert(out.data != 0);
  assert(in.data != 0);
  assert(in.nCols == out.nCols);
  assert(in.nRows == out.nRows);
  dim3 size(in.nRows, in.nCols, in.nChannels);  
  assert(out.nChannels % in.nChannels == 0);
  _translate<<<g, b>>>(in.data, out.data, dx, dy, size, out.nChannels / in.nChannels, roundToInt);
  cuda_err_check();
}

__global__ void _center_crop(const IntervalGPU* in, IntervalGPU* out, const dim3 inSize, const dim3 outSize) {
  const size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  const size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  const size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;
  const size_t dx =  (inSize.x - outSize.x) / 2;
  const size_t dy =  (inSize.y - outSize.y) / 2;
  if (row < outSize.x && col < outSize.y && chan < outSize.z) {
    const size_t m = grid_rcc_to_idx(outSize, row, col, chan);
    const size_t n = grid_rcc_to_idx(inSize, row + dx, col + dy, chan);
    out[m] = in[n];
  }
}

void center_cropGPU(const ImageImplGPU& in, const ImageImplGPU& out) {
  assert(in.nCols >= out.nCols);
  assert(in.nRows >= out.nRows);
  assert(in.nChannels == out.nChannels);
  assert(out.data != 0);
  assert(in.data != 0);
  assert(in.device == out.device);
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(in.device == dev);

  
  const size_t block_z = out.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) out.nRows / (float) block_xy), ceilf( (float) out.nCols / (float) block_xy), 1);
  dim3 inSize(in.nRows, in.nCols, in.nChannels);
  dim3 outSize(out.nRows, out.nCols, out.nChannels);
  _center_crop<<<g, b>>>(in.data, out.data, inSize, outSize);
  cuda_err_check();
}


__global__ void _filter_vignetting(const IntervalGPU* in, IntervalGPU* out, const float* filter,
                                   const dim3 size, const size_t c, const float radiusSq) {
  const size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  const size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  const size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;
  if (row < size.x && col < size.y && chan < size.z) {  
    const int x =  2 * col - (size.y - 1);
    const int y =  2 * row - (size.x - 1);
    if (x*x + y*y > radiusSq) {
      out[grid_rcc_to_idx(size, row, col, chan)] = d_make_IntervalGPU(0, 0);
    }
    else {
      IntervalGPU tmp = d_make_IntervalGPU(0, 0);
      for (int i = -c; i <= signed(c); ++i)
        for (int j = -c; j <= signed(c); ++j) {
          const int r_ = row + i;
          const int c_ = col + j;
          const int x_ =  2 * c_ - (size.y - 1);
          const int y_ =  2 * r_ - (size.x - 1);          
          if (r_ > 0 && r_ < signed(size.x) && c_ > 0 && c_ < signed(size.y) &&
              x_*x_ + y_*y_ <= radiusSq) {
            tmp += filter[abs(i) * (c+1) +  abs(j)] * in[grid_rcc_to_idx(size, r_, c_, chan)];
          }
        }
      assert(tmp.inf <= tmp.sup + Constants::EPS);
      const size_t m = grid_rcc_to_idx(size, row, col, chan);
      out[m] = tmp;
    }
  }
}


void filter_vignettingGPU(const ImageImplGPU& in, const ImageImplGPU& out, const float* filter,
                          const size_t c, const float radiusSq) {
  assert(in.nCols == out.nCols);
  assert(in.nRows == out.nRows);
  assert(in.nChannels == out.nChannels);
  assert(out.data != 0);
  assert(in.data != 0);
  assert(filter != 0);
  assert(in.device == out.device);
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(in.device == dev);
  
  const size_t block_z = out.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) out.nRows / (float) block_xy), ceilf( (float) out.nCols / (float) block_xy), 1);
  dim3 size(in.nRows, in.nCols, in.nChannels);
    _filter_vignetting<<<g, b>>>(in.data, out.data, filter, size, c, radiusSq);
    cuda_err_check();
  }


  __global__ void _vignetting(const IntervalGPU* in, IntervalGPU* out,
                              const dim3 size, const float radiusSq) {
    const size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
    const size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
    const size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;
    if (row < size.x && col < size.y && chan < size.z) {
      const int x =  2 * col - (size.y - 1);
      const int y =  2 * row - (size.x - 1);
      if (x*x + y*y > radiusSq) {
        out[grid_rcc_to_idx(size, row, col, chan)] = d_make_IntervalGPU(0, 0);
      }
      else {
        const size_t m = grid_rcc_to_idx(size, row, col, chan);
        out[m] = in[m];          
      }
    }
  }


  void vignettingGPU(const ImageImplGPU& in, const ImageImplGPU& out, const float radiusSq) {
    assert(in.nCols == out.nCols);
    assert(in.nRows == out.nRows);
    assert(in.nChannels == out.nChannels);
    assert(out.data != 0);
    assert(in.data != 0);
    assert(in.device == out.device);
    int dev;
    cuda_err_print(cudaGetDevice(&dev));
    assert(in.device == dev);
  
    const size_t block_z = out.nChannels;
    const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
    dim3 b(block_xy, block_xy, block_z);
    dim3 g(ceilf( (float) out.nRows / (float) block_xy), ceilf( (float) out.nCols / (float) block_xy), 1);
    dim3 size(in.nRows, in.nCols, in.nChannels);
    _vignetting<<<g, b>>>(in.data, out.data, size, radiusSq);
    cuda_err_check();
  }

  
__global__ void _filterVignettingL2diffCGPU(float *odata,
                                            const IntervalGPU* lhs,
                                            const IntervalGPU* rhs,
                                            const IntervalGPU* vingette,
                                            const dim3 size,
                                            const float radiusSq,
                                            const float* filter,
                                            const size_t c, /*  filter c */
                                            const size_t nrNorms,
                                            const size_t blocksPerNorm
                                            ) {
  extern __shared__ float sdata[];
  const size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  const size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  const size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;
  assert(threadIdx.z == 0); // always only one channel per block
  const size_t tid = threadIdx.x*blockDim.y + threadIdx.y;
  const size_t bid = blockIdx.z*gridDim.x*gridDim.y + blockIdx.x * gridDim.y + blockIdx.y;
  const size_t blockSize = blockDim.x*blockDim.y*blockDim.z;
  assert(tid < blockSize);
  sdata[tid] = 0;
  
  float res = 0;
  if (row < size.x && col < size.y && chan < size.z) {
    const int x =  2 * col - (size.y - 1);
    const int y =  2 * row - (size.x - 1);

    if (x*x + y*y <= radiusSq) {
      if (filter != 0) {
        IntervalGPU tmp1 = d_make_IntervalGPU(0, 0);
        IntervalGPU tmp2 = d_make_IntervalGPU(0, 0);
        for (int i = -c; i <= signed(c); ++i)
          for (int j = -c; j <= signed(c); ++j) {
            const int r_ = row + i;
            const int c_ = col + j;
            const int x_ =  2 * c_ - (size.y - 1);
            const int y_ =  2 * r_ - (size.x - 1);
            if (r_ > 0 && r_ < signed(size.x) && c_ > 0 && c_ < signed(size.y)
                && x_*x_ + y_*y_ <= radiusSq
                ) {
              const size_t m = grid_rcc_to_idx(size, r_, c_, chan);
              if (vingette == 0 || vingette[m].sup == 0) {
                tmp1 += filter[abs(i) * (c+1) +  abs(j)] * lhs[m];
                tmp2 += filter[abs(i) * (c+1) +  abs(j)] * rhs[m];
              }              
            }
          }
        tmp1 = tmp1 - tmp2;
        res += max(tmp1.inf * tmp1.inf, tmp1.sup * tmp1.sup);      
      } else {
        const size_t m = grid_rcc_to_idx(size, row, col, chan);
        if (vingette == 0 || vingette[m].sup == 0) {
          IntervalGPU tmp = lhs[m] - rhs[m];
          res += max(tmp.inf * tmp.inf, tmp.sup * tmp.sup);
        }
      }
    }
  }
  sdata[tid] = res;
  __syncthreads();
  assert(blockSize >= 64);
  assert(blockSize <= 1024);  
  if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
  if (blockSize >= 64) { if (tid <  32) { sdata[tid] += sdata[tid +  32]; } __syncthreads(); }
  if (blockSize >= 32) { if (tid <  16) { sdata[tid] += sdata[tid +  16]; } __syncthreads(); }
  if (blockSize >= 16) { if (tid <  8) { sdata[tid] += sdata[tid +  8]; } __syncthreads(); }
  if (tid == 0) {
    size_t r = min(blockSize, (size_t)8);
    for(size_t k = 1; k < r; ++k) {
      sdata[0] += sdata[k];
    }
    odata[bid] = sdata[0];
  }
}


vector<float> filterVignettingL2diffCGPU(const ImageImplGPU& lhs,
                                         const ImageImplGPU& rhs,
                                         const ImageImplGPU* vingette,
                                         const float* filter,
                                         const size_t filter_c,
                                         const float radiusSq,
                                         const size_t c) {
  assert(lhs.data != 0);
  assert(rhs.data != 0);
  assert(lhs.nRows == rhs.nRows);
  assert(lhs.nCols == rhs.nCols);
  assert(lhs.nChannels == rhs.nChannels);
  assert(c <= lhs.nChannels);
  assert(lhs.nChannels % c == 0);

  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(lhs.device == dev);
  assert(lhs.device == rhs.device);

  if (vingette != 0) {
    assert(vingette->device == dev);
    assert(lhs.nRows == vingette->nRows);
    assert(lhs.nCols == vingette->nCols);
    assert(lhs.nChannels == vingette->nChannels);
    assert(vingette->data != 0);
  }
  
  const size_t nrNorms = lhs.nChannels / c;
  const dim3 size(lhs.nRows, lhs.nCols, lhs.nChannels);
  const size_t block_z = 1;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z)) / 2;
  dim3 b(block_xy, block_xy, block_z);
  const size_t sx = ceilf( (float) lhs.nRows / (float) block_xy);
  const size_t sy = ceilf( (float) lhs.nCols / (float) block_xy);
  dim3 g(sx, sy, lhs.nChannels); 
  //there are sx*sy blocks per channel
  size_t blocksPerNorm = sx*sy * c;
  float* parts;
  cuda_err_print(cudaMalloc((void**)&parts, sx*sy*lhs.nChannels*sizeof(float*)));
  cuda_err_print(cudaMemset(parts, 0, sx*sy*lhs.nChannels*sizeof(float*)));
  _filterVignettingL2diffCGPU<<<g, b, block_xy*block_xy*block_z*sizeof(float)>>>(parts,
                                                                                 lhs.data,
                                                                                 rhs.data,
                                                                                 (vingette == 0) ? 0 : vingette->data,
                                                                                 size,
                                                                                 radiusSq,
                                                                                 filter,
                                                                                 filter_c,
                                                                                 nrNorms,
                                                                                 blocksPerNorm);
  cuda_err_check();
  
  const size_t count = c * sx * sy;
  const size_t n = min((size_t) pow(2, ceilf(log2(count))), (size_t)1024); //find smallest power of two larger than count
  const size_t workPerThread = (size_t)ceilf((float)count/n);
  float *norms = 0;
  
  cuda_err_print(cudaMalloc((void**)&norms, nrNorms*sizeof(float)));
  reduce<<<nrNorms, n, n*sizeof(float)>>>(parts, norms, n, count, workPerThread);
  cuda_err_print(cudaFree(parts));

  vector<float> normsOut(nrNorms);
  cuda_err_print(cudaMemcpy(&normsOut[0], norms, nrNorms*sizeof(float), cudaMemcpyDeviceToHost));
  cuda_err_print(cudaFree(norms));
  return normsOut;
}


__global__ void _rect_vignetting(IntervalGPU* img, const dim3 size, const size_t filter_size) {

  const size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  const size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  const size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;

  
  if (row < size.x && col < size.y && chan < size.z) {

    bool flip = false;
    if (row < filter_size) {flip = true;}
    else if (row >= size.x - filter_size) { flip = true;}
    else {
      flip = col < filter_size || col >= size.y - filter_size;
    }
    if (flip) {
      const size_t m = grid_rcc_to_idx(size, row, col, chan);
      img[m] = d_make_IntervalGPU(0, 0);
    }  
  }
}

void rect_vignettingGPU(ImageImplGPU& img, const size_t filter_size) {
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(img.device == dev);
  const size_t block_z = img.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z)) / 2;
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) img.nRows / (float) block_xy), ceilf( (float) img.nCols / (float) block_xy), 1);
  assert(img.data != 0);
  dim3 size(img.nRows, img.nCols, img.nChannels);  
  _rect_vignetting<<<g, b>>>(img.data, size, filter_size);
  cuda_err_check();

}

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
                const bool computeVingette) {
  assert(isInit);
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(in.device == dev);
  const size_t block_xy = floorf(sqrt(1024.0)) / 2;
  dim3 b(block_xy, block_xy, 1);
  dim3 g(ceilf( (float) in.nRows / (float) block_xy), ceilf( (float) in.nCols / (float) block_xy), 1);
  assert(out != 0);
  if (computeVingette) {
    assert(vingette != 0);
    *vingette = new ImageImplGPU(in.nRows, in.nCols, in.nChannels, in.device);
  }
  ImageImplGPU *ret = new ImageImplGPU(in.nRows, in.nCols, in.nChannels, in.device);
  ImageImplGPU *constraints = 0;  
  dim3 size(in.nRows, in.nCols, in.nChannels);
  bool *d_isEmpty = 0, *d_isEmptyCombined = 0;
  size_t blockSize = b.x * b.y * b.z;
  size_t gridSize = g.x * g.y * g.z;
  size_t s_isEmpty = 16*gridSize * sizeof(bool);
  cuda_err_print(cudaMalloc((void**)&d_isEmpty, s_isEmpty));
  size_t G = ceil(s_isEmpty/1024.);  
  cuda_err_print(cudaMalloc((void**)&d_isEmptyCombined, 8*G* sizeof(bool)));
  bool *h_isEmpty = 0;
  cuda_err_print(cudaMallocHost((void**)&h_isEmpty, 8*G* sizeof(bool)));  
  bool isEmpty = false;
  for(size_t it = 0; it < (nrRefinements + 1); ++it) {
    bool refine = it > 0;
    bool lastIt = (it == nrRefinements);

    if (refine) {
      ImageImplGPU *tmp = ret;
      ret = constraints;
      constraints = tmp;
      if (ret == 0) ret = new ImageImplGPU(in.nRows, in.nCols, in.nChannels, in.device);
    }  
    switch(in.nChannels) {
    case 1:
      _invert_pixel_on_bounding_box<1><<<g, b, blockSize*sizeof(bool)>>>(in.data,
                                                                         ret->data,
                                                                         d_isEmpty,
                                                                         size,
                                                                         refine,
                                                                         (constraints == 0) ? 0 : constraints->data,
                                                                         transform,
                                                                         transform_param0, // cgamma for rot, dx for trans
                                                                         transform_param1, // sgamma for rot, dy for trans
                                                                         addIntegerError,
                                                                         lastIt && toIntegerValues,
                                                                         lastIt && computeVingette,
                                                                         (lastIt && computeVingette) ? (*vingette)->data : 0);      
      break;
    case 3:
      _invert_pixel_on_bounding_box<3><<<g, b, blockSize*sizeof(bool)>>>(in.data,
                                                                         ret->data,
                                                                         d_isEmpty, size,
                                                                         refine,
                                                                         (constraints == 0) ? 0 : constraints->data,
                                                                         transform,
                                                                         transform_param0, // cgamma for rot, dx for trans
                                                                         transform_param1, // sgamma for rot, dy for trans
                                                                         addIntegerError,
                                                                         lastIt && toIntegerValues,
                                                                         lastIt && computeVingette,
                                                                         (lastIt && computeVingette) ? (*vingette)->data : 0);
      break;
    default: assert(false);
    };
    if(lastIt || stopEarly) {
      _combineEmpty<<<G, 1024, 1024*sizeof(bool)>>>(d_isEmpty, d_isEmptyCombined, s_isEmpty);
      cuda_err_print(cudaMemcpy(h_isEmpty, d_isEmptyCombined, G*sizeof(bool), cudaMemcpyDeviceToHost));
      for(size_t i = 0; i < G; ++i) {
        isEmpty = isEmpty || h_isEmpty[i];
      }
      if (stopEarly && isEmpty) break;
    }
  }
  if (constraints != 0) delete constraints;
  *out = ret;
  cuda_err_print(cudaFree((void*)d_isEmpty));
  cuda_err_print(cudaFree((void*)d_isEmptyCombined));
  cuda_err_print(cudaFreeHost((void*)h_isEmpty));
  return isEmpty;
}




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
                                                  IntervalGPU* vingette) {
    extern __shared__ bool isEmpty[];
    assert(C == size.z);
    assert(constraints != out);
    const size_t nRow =  blockIdx.x*blockDim.x + threadIdx.x;
    const size_t nCol =  blockIdx.y*blockDim.y + threadIdx.y;

    const IntervalGPU zeroOne = d_make_IntervalGPU(0, 1);
    const IntervalGPU empty = d_make_empty_IntervalGPU();
    const IntervalGPU zero = d_make_IntervalGPU(0, 0);
    const IntervalGPU one = d_make_IntervalGPU(1, 1);
  
    const size_t blockSize = blockDim.x * blockDim.y;
    const size_t tid = threadIdx.x*blockDim.y + threadIdx.y;
    const size_t bid = blockIdx.x * gridDim.y + blockIdx.y;
    isEmpty[tid] = false; 
  
    if (nRow < size.x && nCol < size.y) {
      const float2 coord = getCoordinate(size, nRow, nCol);
      const IntervalGPU coordI_x = d_make_IntervalGPU(coord.x - 2, coord.x + 2);
      const IntervalGPU coordI_y = d_make_IntervalGPU(coord.y - 2, coord.y + 2);
      IntervalGPU coordI_x_pre, coordI_y_pre;
    
      if (transform == Transform::rotation) {
        //transform_param0 == cgamma
        //transform_param1 == sgamma
        coordI_x_pre = coordI_x * transform_param0 + coordI_y * transform_param1;
        coordI_y_pre = -coordI_x * transform_param1 + coordI_y * transform_param0;
      } else if (transform == Transform::translation) {
        //transform_param0 == dx
        //transform_param1 == dy
        coordI_x_pre = coordI_x  - 2 * transform_param0;
        coordI_y_pre = coordI_y - 2 * transform_param1;      
      } else {
        assert(false);
      }

       
      // pixels to consider
      size_t parity_x = (size.y - 1) % 2;
      size_t parity_y = (size.x - 1) % 2;
      int lo_x, hi_x, lo_y, hi_y;
      int4 bounds = calculateBoundingBox(coordI_x_pre, coordI_y_pre, parity_x, parity_y);
      lo_x = bounds.x;
      hi_x = bounds.y;
      lo_y = bounds.z;
      hi_y = bounds.w;
    
      IntervalGPU inv_p[C];
      for (size_t chan = 0; chan < C; ++chan) {
        inv_p[chan] = zeroOne;
      }
    
      for (int x_iter = lo_x; x_iter <= hi_x; x_iter += 2) 
        for (int y_iter = lo_y; y_iter <= hi_y; y_iter += 2) {
          IntervalGPU coordI_x_iterT, coordI_y_iterT;
          if (transform == Transform::rotation) {
            //transform_param0 == cgamma
            //transform_param1 == sgamma
            coordI_x_iterT = x_iter * transform_param0 - y_iter * transform_param1;
          coordI_y_iterT = x_iter * transform_param1 + y_iter * transform_param0;
        } else if (transform == Transform::translation) {
          //transform_param0 == dx
          //transform_param1 == dy
          coordI_x_iterT = x_iter + 2 * transform_param0;
          coordI_y_iterT = y_iter + 2 * transform_param1; 
        } else {
          assert(false);
        }

        if (intervalGPU_is_empty(meet(coordI_x_iterT, coordI_x)) ||
            intervalGPU_is_empty(meet(coordI_y_iterT, coordI_y))) continue;
        
        IntervalGPU p[C];
        int2 rc_iter = coord_to_rc(size, x_iter, y_iter);
        if (0 <= rc_iter.x && rc_iter.x < size.x &&
            0 <= rc_iter.y && rc_iter.y < size.y) {
          for (size_t chan = 0; chan < C; ++chan) {
            p[chan] = valueAt(in, size, x_iter, y_iter, chan, zeroOne);
            
            if (addIntegerError) {
              //this assumes that the pixel_values * 255 are integers in [0, 255]
              //and that the rounding mode is arithmetic rounding (and not for example floorfing)
              assert(p[chan].inf == p[chan].sup);
              float deltaUp = (0.5f - Constants::EPS)/255.0f;
              float deltaDown = 0.5f / 255.0f;
              p[chan] = meet(d_make_IntervalGPU(p[chan].inf - deltaDown, p[chan].sup + deltaUp), zeroOne);
            }    
          }
        } else {
          for (size_t chan = 0; chan < C; ++chan)
            p[chan] = zeroOne;
        }

        
        IntervalGPU inv_corners[C];
        for (size_t chan = 0; chan < C; ++chan)
          inv_corners[chan] = empty;
        for (Corner c : {Corner::upper_left, Corner::lower_left, Corner::upper_right, Corner::lower_right}) {
          IntervalGPU inv_corner[C];
          _invert_pixel<C>(c,
                           size,
                           coord,
                           coordI_x_iterT,
                           coordI_y_iterT,
                           inv_corner,
                           p,
                           constraints,
                           refine,
                           false); //debug          
          for (size_t chan = 0; chan < C; ++chan) {
            inv_corners[chan] = join(inv_corners[chan], inv_corner[chan]);
          }
        }
        for (size_t chan = 0; chan < C; ++chan) {
          inv_p[chan] = meet(inv_p[chan], inv_corners[chan]);
        }
      }

    for (size_t chan = 0; chan < C; ++chan) {
      const size_t m = grid_rcc_to_idx(size, nRow, nCol, chan);
      if (toIntegerValues && !intervalGPU_is_empty(inv_p[chan])) {
        float lower = inv_p[chan].inf;
        float upper = inv_p[chan].sup;
    

        if (lower > 0) {
          for(size_t k = 0; k < 255; k ++) {
            if(d_pixelVals[k] < lower && lower <= d_pixelVals[k+1] ) {
              lower = d_pixelVals[k+1];
              break;
            }
          }
        }

        assert(lower >= inv_p[chan].inf);
          
        if(upper < 1) {
          for(size_t k = 256; k >= 1; k--) {
            if(d_pixelVals[k] > upper && upper >= d_pixelVals[k]) {
              upper = d_pixelVals[k];
              break;
            }
          }
        }
        assert(upper <= inv_p[chan].sup);
          
          
        if (lower > upper) {
          inv_p[chan] = empty;
        } else {
          inv_p[chan] = d_make_IntervalGPU(lower, upper);
        }
      }
      out[m] = inv_p[chan];

      if (computeVingette) {        
        if ((out[m].sup - out[m].inf) > 0.3f) {
          vingette[m] = one;
        } else {
          vingette[m] = zero;
        }
      }      
      
      isEmpty[tid] = isEmpty[tid] || intervalGPU_is_empty(out[m]);    
    }
  }
    
  assert(blockSize <= 1024);
  __syncthreads();
  if (blockSize >= 512) {if (tid < 512 && tid + 512 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 512]; } __syncthreads();}
  if (blockSize >= 256) {if (tid < 256 && tid + 256 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 256]; } __syncthreads();}
  if (blockSize >= 128) {if (tid < 128 && tid + 128 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 128]; } __syncthreads();}
  if (blockSize >= 64) {if (tid < 64 && tid + 64 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 64]; } __syncthreads();}
  if (blockSize >= 32) {if (tid < 32 && tid + 32 < blockSize ) { isEmpty[tid] = isEmpty[tid] || isEmpty[tid + 32]; } __syncthreads();}


  if (tid < 16) {
    bool e = false;
    if (tid < blockSize) e = e || isEmpty[tid];
    if (tid + 16 < blockSize) e = e || isEmpty[tid + 16];    
    isEmptyMask[bid * 16 + tid] = e;
  }
}

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
                              const bool debug) {
  //this method has a lot of control flow, however this should not be
  //an issue in kernel execution as all threads do the same (unless
  //they go into the early return)
  IntervalGPU x_box, y_box;
  float x_ll, y_ll;

  float2 corners[4];
  size_t c_corners = 0;

  const IntervalGPU zeroOne = d_make_IntervalGPU(0, 1);
  
  switch(center) {
  case lower_right:
    x_ll = coord.x;
    y_ll = coord.y;
    x_box = meet(coord_iterTx, d_make_IntervalGPU(coord.x, coord.x+2));
    y_box = meet(coord_iterTy, d_make_IntervalGPU(coord.y, coord.y+2));
    if (!refine)
      corners[c_corners++] = make_float2(x_box.sup, y_box.sup);
    break;
  case upper_right:
    x_ll = coord.x;
    y_ll = coord.y - 2;
    x_box = meet(coord_iterTx, d_make_IntervalGPU(coord.x, coord.x+2));
    y_box = meet(coord_iterTy, d_make_IntervalGPU(coord.y-2, coord.y));
    if (!refine)
      corners[c_corners++] = make_float2(x_box.sup, y_box.inf);
    break;
  case lower_left:
    x_ll = coord.x - 2;
    y_ll = coord.y;
    x_box = meet(coord_iterTx, d_make_IntervalGPU(coord.x-2, coord.x));
    y_box = meet(coord_iterTy, d_make_IntervalGPU(coord.y, coord.y+2));
    if (!refine)
      corners[c_corners++] = make_float2(x_box.inf, y_box.sup);
    break;
  case upper_left:
    x_ll = coord.x - 2;
    y_ll = coord.y - 2;
    x_box = meet(coord_iterTx, d_make_IntervalGPU(coord.x-2, coord.x));
    y_box = meet(coord_iterTy, d_make_IntervalGPU(coord.y-2, coord.y));
    if (!refine)
      corners[c_corners++] = make_float2(x_box.inf, y_box.inf);
    break;
  default:
    assert(false);

  };

  if (debug) printf("%f %f %f %f %f %f\n", x_ll, y_ll, x_box.inf, x_box.sup, y_box.inf, y_box.sup);
  
  assert(out != 0);
  const IntervalGPU empty = d_make_empty_IntervalGPU();
  for (size_t chan = 0; chan < C; ++chan)
    out[chan] = empty;

  if (intervalGPU_is_empty(x_box) || intervalGPU_is_empty(y_box)) {
    return;
  }

  assert(pixel_values != 0);
    
  if (refine) {
    assert(constraints != 0);
    corners[c_corners++] = make_float2(x_box.inf, y_box.inf);
    corners[c_corners++] = make_float2(x_box.sup, y_box.inf);
    corners[c_corners++] = make_float2(x_box.inf, y_box.sup);
    corners[c_corners++] = make_float2(x_box.sup, y_box.sup);
  }

  assert(c_corners == 1 || c_corners == 4);
  for (size_t i = 0; i < c_corners; ++i){
    float2 coord = corners[i];
    IntervalGPU ret[C];
    for (size_t chan = 0; chan < C; ++chan) ret[chan] = empty;
    float4 coef_ =  bilinear_coefficients(x_ll, y_ll, coord.x, coord.y);
    float* coef = &coef_.x;
    if (debug) printf("coefs %f %f %f %f\n", coef[0], coef[1], coef[2], coef[3]);
    if (coef[center] == 0) {
      for (size_t chan = 0; chan < C; ++chan)
        ret[chan] = zeroOne;
    } else {
      for (size_t chan = 0; chan < C; ++chan) {
        ret[chan] = pixel_values[chan];
        if (debug) printf("[%f %f]", ret[chan].inf, ret[chan].sup); 
      }
      if (debug) printf("\n");

      if(refine) {
        for(size_t k = 0; k < 4; ++k) {
          if (k == center) continue;
          float x_ = 0, y_ = 0;
          switch(k) {
          case 0:
            x_ = x_ll;
            y_ = y_ll;
            break;
          case 1:
            x_ = x_ll;
            y_ = y_ll + 2;
            break;
          case 2:
            x_ = x_ll + 2;
            y_ = y_ll;
            break;
          case 3:
            x_ = x_ll + 2;
            y_ = y_ll + 2;
            break;
          }
          for (size_t chan = 0; chan < C; ++chan) {
            IntervalGPU con = valueAt(constraints, size, x_, y_, chan, zeroOne);
            ret[chan] = ret[chan] - coef[k] * con; 
          }
        }
      } else {
        float f = 0;
        for(size_t k = 0; k < 4; ++k) {
          if (k == center) continue;
          f += coef[k];
        }
        IntervalGPU tmp = f * zeroOne;
        for (size_t chan = 0; chan < C; ++chan)
          ret[chan] = ret[chan] - tmp;
      }

      for (size_t chan = 0; chan < C; ++chan)
        ret[chan] = ret[chan] / coef[center];
    }
    for (size_t chan = 0; chan < C; ++chan) {
      out[chan] = join(out[chan], ret[chan]);
    }
  }
}



__global__ void _customVingette(IntervalGPU* out, const IntervalGPU* in, const IntervalGPU* vingette, const dim3 size) {
  const size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  const size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  const size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;

  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    if (vingette[m].sup == 0) {
      out[m] = in[m];
    } else {
      out[m] = d_make_IntervalGPU(0, 0);
    }
  }
}


void customVingetteGPU(ImageImplGPU& out, const ImageImplGPU& in, const ImageImplGPU& vingette) {
  assert(in.device == out.device);
  assert(in.device == vingette.device);
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(in.device == dev);
  assert(out.data != 0);
  assert(in.data != 0);
  assert(vingette.data != 0);
  assert(in.nCols == out.nCols);
  assert(in.nRows == out.nRows);
  assert(in.nChannels == out.nChannels);
  assert(in.nCols == vingette.nCols);
  assert(in.nRows == vingette.nRows);
  assert(in.nChannels == vingette.nChannels);
  
  const size_t block_z = in.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z)) / 2;
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) in.nRows / (float) block_xy), ceilf( (float) in.nCols / (float) block_xy), 1); 
  dim3 size(in.nRows, in.nCols, in.nChannels);
  _customVingette<<<g, b>>>(out.data, in.data, vingette.data, size);
  cuda_err_check();
}

__global__ void _threshold(IntervalGPU* out, const IntervalGPU* in, const dim3 size, const float val) {
  size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;    
  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    if (!intervalGPU_is_empty(in[m]) && in[m].inf >= val) {
      out[m] = d_make_IntervalGPU(1, 1);
    } else {
      out[m] = d_make_IntervalGPU(0, 0);
    }    
  }
}

void thresholdGPU(ImageImplGPU& out, const ImageImplGPU& in, const float val) {
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(in.device == dev);
  assert(in.device == out.device);
  assert(in.nRows == out.nRows);
  assert(in.nCols == out.nCols);
  assert(in.nChannels == out.nChannels);
  const size_t block_z = in.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) in.nRows / (float) block_xy), ceilf( (float) in.nCols / (float) block_xy), 1);
  assert(in.data != 0);
  dim3 size(in.nRows, in.nCols, in.nChannels);
  _threshold<<<g, b>>>(out.data, in.data, size, val);
  cuda_err_check();
}


__global__ void _pixelOr(IntervalGPU* out, const IntervalGPU* lhs, const IntervalGPU* rhs, const dim3 size) {
  size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;    
  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    out[m] = (lhs[m].inf > 0 || rhs[m].inf > 0) ? d_make_IntervalGPU(1, 1) : d_make_IntervalGPU(0, 0);
  }
}


void pixelOrGPU(ImageImplGPU& out, const ImageImplGPU& lhs, const ImageImplGPU& rhs) {
  assert(lhs.device == rhs.device);
  assert(lhs.device == out.device);
  int dev;
  cudaGetDevice(&dev);
  assert(lhs.device == dev);
  const size_t block_z = out.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) out.nRows / (float) block_xy), ceilf( (float) out.nCols / (float) block_xy), 1);
  assert(out.data != 0);
  assert(lhs.data != 0);
  assert(rhs.data != 0);
  assert(out.nRows == rhs.nRows);
  assert(out.nCols == rhs.nCols);
  assert(out.nChannels == rhs.nChannels);
  assert(out.nRows == lhs.nRows);
  assert(out.nCols == lhs.nCols);
  assert(out.nChannels == lhs.nChannels);
  dim3 size(out.nRows, out.nCols, out.nChannels);
  _pixelOr<<<g, b>>>(out.data, lhs.data, rhs.data, size);
  cuda_err_check();
}



__global__ void _pixelAnd(IntervalGPU* out, const IntervalGPU* lhs, const IntervalGPU* rhs, const dim3 size) {
  size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;    
  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    out[m] = (lhs[m].inf > 0 && rhs[m].inf > 0) ? d_make_IntervalGPU(1, 1) : d_make_IntervalGPU(0, 0);
  }
}


void pixelAndGPU(ImageImplGPU& out, const ImageImplGPU& lhs, const ImageImplGPU& rhs) {
  assert(lhs.device == rhs.device);
  assert(lhs.device == out.device);
  int dev;
  cudaGetDevice(&dev);
  assert(lhs.device == dev);
  const size_t block_z = out.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) out.nRows / (float) block_xy), ceilf( (float) out.nCols / (float) block_xy), 1);
  assert(out.data != 0);
  assert(lhs.data != 0);
  assert(rhs.data != 0);
  assert(out.nRows == rhs.nRows);
  assert(out.nCols == rhs.nCols);
  assert(out.nChannels == rhs.nChannels);
  assert(out.nRows == lhs.nRows);
  assert(out.nCols == lhs.nCols);
  assert(out.nChannels == lhs.nChannels);
  dim3 size(out.nRows, out.nCols, out.nChannels);
  _pixelAnd<<<g, b>>>(out.data, lhs.data, rhs.data, size);
  cuda_err_check();
}




__global__ void _erode(IntervalGPU* out, const IntervalGPU* in, const dim3 size) {
  size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;    
  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);

    IntervalGPU val = in[m];
    for(int dr=-1; dr <= 1; ++dr)
      for(int dc=-1; dc <= 1; ++dc) {
        int r = (signed)row + dr;
        int c = (signed)col + dc;
        if (0 <= r && r < size.x && 0 <= c && c < size.y) {
          const size_t mm = grid_rcc_to_idx(size, (unsigned)r, (unsigned)c, chan);
          IntervalGPU other = in[mm];
          if (other.inf <= val.inf && other.sup <= val.sup)
            val = other;
        }
      }
    out[m] = val;
  }
}



void erodeGPU(ImageImplGPU& out, const ImageImplGPU& in) {
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(in.device == dev);
  assert(in.device == out.device);
  assert(in.nRows == out.nRows);
  assert(in.nCols == out.nCols);
  assert(in.nChannels == out.nChannels);
  const size_t block_z = in.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z));
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) in.nRows / (float) block_xy), ceilf( (float) in.nCols / (float) block_xy), 1);
  assert(in.data != 0);
  dim3 size(in.nRows, in.nCols, in.nChannels);
  _erode<<<g, b>>>(out.data, in.data, size);
  cuda_err_check();
}


__global__ void _widthVingette(IntervalGPU* out, const IntervalGPU* in, const IntervalGPU* vingette, const dim3 size, const float treshold) {
  const size_t row =  blockIdx.x*blockDim.x + threadIdx.x;
  const size_t col =  blockIdx.y*blockDim.y + threadIdx.y;
  const size_t chan =  blockIdx.z*blockDim.z + threadIdx.z;

  if (row < size.x && col < size.y && chan < size.z) {
    const size_t m = grid_rcc_to_idx(size, row, col, chan);
    if ((vingette[m].sup - vingette[m].inf) <= treshold) {
      out[m] = in[m];
    } else {
      out[m] = d_make_IntervalGPU(0, 0);
    }
  }
}


void widthVingetteGPU(ImageImplGPU& out, const ImageImplGPU& in, const ImageImplGPU& vingette, const float treshold) {
  assert(in.device == out.device);
  assert(in.device == vingette.device);
  int dev;
  cuda_err_print(cudaGetDevice(&dev));
  assert(in.device == dev);
  assert(out.data != 0);
  assert(in.data != 0);
  assert(vingette.data != 0);
  assert(in.nCols == out.nCols);
  assert(in.nRows == out.nRows);
  assert(in.nChannels == out.nChannels);
  assert(in.nCols == vingette.nCols);
  assert(in.nRows == vingette.nRows);
  assert(in.nChannels == vingette.nChannels);
  
  const size_t block_z = in.nChannels;
  const size_t block_xy = floorf(sqrt(1024.0/ (float)block_z)) / 2;
  dim3 b(block_xy, block_xy, block_z);
  dim3 g(ceilf( (float) in.nRows / (float) block_xy), ceilf( (float) in.nCols / (float) block_xy), 1); 
  dim3 size(in.nRows, in.nCols, in.nChannels);
  _widthVingette<<<g, b>>>(out.data, in.data, vingette.data, size, treshold);
  cuda_err_check();
}
