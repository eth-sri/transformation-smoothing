#include "gpu/image_gpu.h"
#include "utils/image_cpu.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "gpu/gpu.h"

using namespace std;

/*** constuctors and friends ***/

ImageImplGPU::ImageImplGPU(size_t nRows, size_t nCols, size_t nChannels, size_t device) : device(device) {
  this->nRows = nRows;
  this->nCols = nCols;
  this->nChannels = nChannels;
  cuda_err_print(cudaSetDevice(device));
  cuda_err_print(cudaMalloc((void **)&this->data, nRows * nCols * nChannels * sizeof(IntervalGPU)));
  //cout << nRows * nCols * nChannels * sizeof(IntervalGPU) /1024/1024 << "MB" << endl;
}

void ImageImplGPU::copyFrom(const ImageImpl &other) {
  const size_t s = other.nRows * other.nCols * other.nChannels;
  auto other_ptr = &other;
  if (other_ptr->isGPU()) {
    const ImageImplGPU& otherGPU = dynamic_cast<const ImageImplGPU&>(other);

    if(otherGPU.device == device) {
      cudaMemcpy(data, otherGPU.data, s * sizeof(IntervalGPU), cudaMemcpyDeviceToDevice);
    } else {
      assert(otherGPU.device != device);    
      cuda_err_print(cudaMemcpyPeer(data, device, otherGPU.data, otherGPU.device, s * sizeof(IntervalGPU))); 
    }
    

  } else {
    IntervalGPU *tmp = new IntervalGPU[s]; //this needs to be dynamic allocation as it might be too large on iamgenet else (too large for stack)
    for(size_t i = 0; i < s; ++i) {
         tmp[i].inf = other.a[i].inf;
         tmp[i].sup = other.a[i].sup;
    }  
    cuda_err_print(cudaSetDevice(device));
    cuda_err_print(cudaMemcpy(data, tmp, s * sizeof(IntervalGPU), cudaMemcpyHostToDevice));
    delete[] tmp;
        
  }
}


ImageImplGPU::ImageImplGPU(const ImageImpl &other, const size_t device) : device(device) {
  this->nRows = other.nRows;
  this->nCols = other.nCols;
  this->nChannels = other.nChannels;
  const size_t s = nRows * nCols * nChannels;  
  cuda_err_print(cudaSetDevice(device));
  cuda_err_print(cudaMalloc((void **)&this->data, s * sizeof(IntervalGPU)));
  //cout << s * sizeof(IntervalGPU) /1024/1024 << "MB" << endl;
  copyFrom(other);
}

ImageImpl& ImageImplGPU::operator = (const ImageImpl &other) {
  assert(false);
}


ImageImplGPU::~ImageImplGPU() {
  if (this->data != 0) {
    cuda_err_print(cudaSetDevice(device));
    cuda_err_print(cudaFree(this->data));
    //cout << "free" <<endl;
    this->data = 0;
      
  }
}

// ImageImplGPU::ImageImplGPU(const &ImageImpl) {};


ImageImpl* ImageImplGPU::resize(const size_t new_nRows, const size_t new_nCols, const bool roundToInt) const {
  return this->resize(new_nRows, new_nCols, roundToInt, 0);
}

ImageImpl* ImageImplGPU::resize(const size_t new_nRows, const size_t new_nCols, const bool roundToInt, const size_t size) const {
  cuda_err_print(cudaSetDevice(device));
    ImageImplGPU *ret;
    size_t dx, dy;
    if (size > 0) {
      ret = new ImageImplGPU(size, size, this->nChannels, this->device);
      assert(new_nRows >= size);
      assert(new_nCols >= size);
      assert(size % 2 == 0);
      //assert(new_nRows % 2 == 0);
      //assert(new_nCols % 2 == 0);
      dx =  (new_nRows - size) / 2;
      dy =  (new_nCols - size) / 2;
    } else {
      ret = new ImageImplGPU(new_nRows, new_nCols, this->nChannels, this->device);
      dx = 0;
      dy = 0;
    }
  
    resizeCropGPU(*this, *ret, roundToInt, new_nRows, new_nCols, dx, dy);  
    return ret;
}



bool ImageImplGPU::inverseRot(ImageImpl** out_ret,
                              ImageImpl** out_vingette,
                              const Interval gamma,
                              const size_t refinements,
                              const bool addIntegerError,
                              const bool toIntegerValues,
                              const bool computeVingette) const {
  cuda_err_print(cudaSetDevice(device));

  const Interval cgamma = cos(gamma);
  const Interval sgamma = sin(gamma);
  const IntervalGPU cgammaG = {cgamma.inf, cgamma.sup};
  const IntervalGPU sgammaG = {sgamma.inf, sgamma.sup};

  assert(out_ret != 0);
  if(computeVingette) assert(out_vingette != 0);

  ImageImplGPU* out_retG = 0;
  ImageImplGPU* out_vingetteG = 0;

  
  bool isEmpty = inverseGPU(*this,
                            &out_retG,
                            &out_vingetteG,
                            Transform::rotation,
                            cgammaG,
                            sgammaG,
                            addIntegerError,
                            toIntegerValues,
                            refinements,
                            true, //stopEarly
                            computeVingette
                            );
  assert(out_retG != 0);
  *out_ret = (ImageImpl*) out_retG;
  if(computeVingette) {
    assert(out_vingetteG != 0);
    *out_vingette = (ImageImpl*) out_vingetteG;
  }
    
  return isEmpty;
}

bool ImageImplGPU::inverseTranslation(ImageImpl** out_ret,
                                      ImageImpl** out_vingette,
                                      const Interval dx,
                                      const Interval dy,
                                      const size_t refinements,
                                      const bool addIntegerError,
                                      const bool toIntegerValues,
                                      const bool computeVingette) const {

  cuda_err_print(cudaSetDevice(device));
  const IntervalGPU dxG = {dx.inf, dx.sup};
  const IntervalGPU dyG = {dy.inf, dy.sup};
  
  assert(out_ret != 0);
  if(computeVingette) assert(out_vingette != 0);  


  ImageImplGPU* out_retG = 0;
  ImageImplGPU* out_vingetteG = 0;

  bool isEmpty = inverseGPU(*this,
                            &out_retG,
                            &out_vingetteG,
                            Transform::translation,
                            dxG,
                            dyG,
                            addIntegerError,
                            toIntegerValues,
                            refinements,
                            true, //stopEarly
                            computeVingette
                            );

  assert(out_retG != 0);
  *out_ret = (ImageImpl*) out_retG;
  if(computeVingette) {
    assert(out_vingetteG != 0);
    *out_vingette = (ImageImpl*) out_vingetteG;
  }
  return isEmpty;
}

void ImageImplGPU::copyTo(Interval* other) const {
  //assert(false); // just to make sure this is not called when debugging memcpy
  cuda_err_print(cudaSetDevice(device));
  const size_t s = nRows * nCols *nChannels;
  IntervalGPU *tmp = new IntervalGPU[s];
  cuda_err_print(cudaMemcpy(tmp, data, s * sizeof(IntervalGPU), cudaMemcpyDeviceToHost));
  for(size_t i = 0; i < s; ++i) {
    if (tmp[i].inf == std::numeric_limits<float>::infinity() && tmp[i].sup == -std::numeric_limits<float>::infinity()) {
      //empty
      other[i] = Interval();
    } else {
      other[i] = {tmp[i].inf, tmp[i].sup}; 
    }
  }
  delete[] tmp;
}

void ImageImplGPU::saveBMP(string fn, bool inf) const {
  assert(false);
}


ImageImplGPU* ImageImplGPU::operator+ (const ImageImplGPU& other) const {
  assert(other.device == device);

  
  cuda_err_print(cudaSetDevice(device));
  ImageImplGPU *ret = new ImageImplGPU(nRows, nCols, nChannels, device);
  addGPU(*ret, *this, other);
    
  return ret;
}


ImageImplGPU* ImageImplGPU::operator- (const ImageImplGPU& other) const {
  assert(other.device == device);

  
  cuda_err_print(cudaSetDevice(device));
  ImageImplGPU *ret = new ImageImplGPU(nRows, nCols, nChannels, device);
  subGPU(*ret, *this, other);
    
  return ret;
}

float ImageImplGPU::l2norm() const {
  return l2normGPU(*this, this->nChannels)[0];
}

vector<float> ImageImplGPU::l2normC(size_t c) const {
  return l2normGPU(*this, c);
}

std::tuple<float, float, float> ImageImplGPU::l2norm_channelwise() const {
  assert(false);
  return {0, 0, 0};
}


void ImageImplGPU::zero() {

  
  cuda_err_print(cudaSetDevice(device));
  
  setGPU(*this, {0, 0});
  
    
}

void ImageImplGPU::roundToInt() {

  
  cuda_err_print(cudaSetDevice(device));
  
  roundToIntGPU(*this);
  
  
}

void ImageImplGPU::clip(const float min_val, const float max_val) {
  cuda_err_print(cudaSetDevice(device));  
  clipGPU(*this, min_val, max_val);
}



//std::vector<ImageImpl*> split_color_channels() const {}
ImageImpl* ImageImplGPU::center_crop(const size_t size) const {
  cuda_err_print(cudaSetDevice(device)); 
  ImageImplGPU *ret = new ImageImplGPU(size, size, this->nChannels, this->device);
  center_cropGPU(*this, *ret);
  return ret;
}

ImageImpl* ImageImplGPU::rotate(const Interval& params, const bool roundToInt) const {
  vector<Interval> p = {params};
  return this->rotate(p, false, roundToInt);
}

ImageImpl* ImageImplGPU::rotate(const vector<Interval> params, const bool singleToMany, const bool roundToInt) const {
  cuda_err_print(cudaSetDevice(device));
  size_t n = params.size();
  assert(n > 0);

  size_t out_chan = this->nChannels;
  vector<Interval> _params( (singleToMany) ? n : out_chan );
  if (singleToMany) {
    out_chan =  this->nChannels * n;
    for(size_t i = 0; i < n; ++i) {
      _params[i] = params[i];
    }
    //}
    //_params.insert( _params.end(), params.begin(), params.end() );
  } else {
    assert(n == 1 || n == this->nChannels);
    if (n == 1) {
      for(size_t i = 0; i < this->nChannels; ++i) {
        _params[i] = params[0];
      }
    } else {
      for(size_t i = 0; i < n; ++i) {
        _params[i] = params[i];
      }
    }
  }

  n = _params.size();
  
  ImageImplGPU *ret = new ImageImplGPU(this->nRows, this->nCols, out_chan, this->device);
  IntervalGPU cgammaGC[n];
  IntervalGPU sgammaGC[n];
  for(size_t i = 0; i < n; ++i) {
    const Interval cgamma = cos(_params[i]);
    const Interval sgamma = sin(_params[i]);
    cgammaGC[i] = {cgamma.inf, cgamma.sup};
    sgammaGC[i] = {sgamma.inf, sgamma.sup};
  }
  IntervalGPU *cgammaGG = 0;
  IntervalGPU *sgammaGG = 0;
  cuda_err_print(cudaMalloc((void**)&cgammaGG, sizeof(IntervalGPU) * n));
  cuda_err_print(cudaMalloc((void**)&sgammaGG, sizeof(IntervalGPU) * n));
  //cout << n * sizeof(IntervalGPU) /1024/1024 << "MB" << endl;
  //cout << n * sizeof(IntervalGPU) /1024/1024 << "MB" << endl;
  cuda_err_print(cudaMemcpy(cgammaGG, cgammaGC, sizeof(IntervalGPU) * n, cudaMemcpyHostToDevice));
  cuda_err_print(cudaMemcpy(sgammaGG, sgammaGC, sizeof(IntervalGPU) * n, cudaMemcpyHostToDevice));
  
  rotateGPU(*this, *ret, cgammaGG, sgammaGG, singleToMany, roundToInt);
  cuda_err_print(cudaFree(sgammaGG));
  cuda_err_print(cudaFree(cgammaGG));  
  
  return ret;
}

void ImageImplGPU::delFilter(float *f) const {
  if (f != 0) {
    cuda_err_print(cudaSetDevice(device));
    cuda_err_print(cudaFree(f));
  }
}

float* ImageImplGPU::getFilter(float sigma, size_t filter_size) const {

  
  cuda_err_print(cudaSetDevice(device));
  
    float * fcpu = ImageImpl::getFilter(sigma, filter_size);
    float * fgpu = 0;
    const size_t c = filter_size / 2;
    cuda_err_print(cudaMalloc((void **)&fgpu, (c+1) * (c+1) * sizeof(float)));
    cuda_err_print(cudaMemcpy(fgpu, fcpu, (c+1) * (c+1) * sizeof(float), cudaMemcpyHostToDevice));
    ImageImpl::delFilter(fcpu);

    
  
    return fgpu;
  }

ImageImpl* ImageImplGPU::filter_vignetting(const float *filter, size_t filter_size, float radiusDecrease) const {
  assert(nRows == nCols);
  assert(filter_size % 2 == 1); // odd filter so we can correctly center it
  
  cuda_err_print(cudaSetDevice(device));
  ImageImplGPU *ret = new ImageImplGPU(nRows, nCols, nChannels, device);
  const bool do_radius = radiusDecrease >= 0;
  const bool do_filter = filter != 0;
  const float radius = nCols - 1 - 2 * radiusDecrease;
  float radiusSq = radius * radius;
  if (!do_radius) radiusSq = std::numeric_limits<float>::infinity();

  if (do_filter) {
    const size_t c = filter_size / 2;
    // use symmetry to only store 1 quarter of the filter
    filter_vignettingGPU(*this, *ret, filter, c, radiusSq);
  } else {
    vignettingGPU(*this, *ret, radiusSq);
  }

    
  
  return ret;
}


vector<float> ImageImplGPU::l2diffC(const ImageImplGPU& other, size_t c) const {
  vector<float> res = l2diffGPU(*this, other, c);
  //cout << res.size() << endl;
  return res;
}


vector<float> ImageImplGPU::filterVignettingL2diffC(const ImageImplGPU& other,
                                                    const ImageImplGPU* vingette,
                                                    const float* filter, size_t filter_size,
                                                    float radiusDecrease, const size_t c) {
  assert(nRows == nCols);
  if (filter != 0) assert(filter_size % 2 == 1); // odd filter so we can correctly center it
  
  cuda_err_print(cudaSetDevice(device));
  const bool do_radius = radiusDecrease >= 0;
  const bool do_filter = filter != 0;
  const float radius = nCols - 1 - 2 * radiusDecrease;
  float radiusSq = radius * radius;
  if (!do_radius) radiusSq = std::numeric_limits<float>::infinity();

  
  const size_t filter_c = filter_size / 2;
  return filterVignettingL2diffCGPU(*this, other, vingette, filter, filter_c, radiusSq, c);
}

ImageImpl* ImageImplGPU::translate(const Interval& dx,
                                   const Interval& dy,
                                   const bool roundToInt) const {
  vector<Interval> dxI = {dx};
  vector<Interval> dyI = {dy};
  return this->translate(dxI, dyI, roundToInt);
}

ImageImpl* ImageImplGPU::translate(const vector<Interval> dx,
                                   const vector<Interval> dy,
                                   const bool roundToInt) const {
  cuda_err_print(cudaSetDevice(device));
  assert(dx.size() == dy.size());
  size_t n = dx.size();
  assert(n > 0);
  // c Channels -> n* c Channels
  IntervalGPU dx_params[n];
  IntervalGPU dy_params[n];
  for(size_t i = 0; i < n; ++i) {
    dx_params[i] = {dx[i].inf, dx[i].sup};
    dy_params[i] = {dy[i].inf, dy[i].sup};
  }

  ImageImplGPU *ret = new ImageImplGPU(this->nRows, this->nCols, n * this->nChannels, this->device);
  IntervalGPU *d_dx = 0, *d_dy = 0;
  cuda_err_print(cudaMalloc((void**)&d_dx, sizeof(IntervalGPU) * n));
  cuda_err_print(cudaMalloc((void**)&d_dy, sizeof(IntervalGPU) * n));
  cuda_err_print(cudaMemcpy(d_dx, dx_params, sizeof(IntervalGPU) * n, cudaMemcpyHostToDevice));
  cuda_err_print(cudaMemcpy(d_dy, dy_params, sizeof(IntervalGPU) * n, cudaMemcpyHostToDevice));

  translateGPU(*this, *ret, d_dx, d_dy, roundToInt);
  cuda_err_print(cudaFree(d_dx));
  cuda_err_print(cudaFree(d_dy));
  return ret;
  }


ImageImpl* ImageImplGPU::customVingette(const ImageImpl* vingette) const {
  assert(vingette->isGPU());
  const ImageImplGPU* vingetteG = (ImageImplGPU*) vingette;
  assert(vingetteG->device == device);
  cuda_err_print(cudaSetDevice(device));
  ImageImplGPU *ret = new ImageImplGPU(nRows, nCols, nChannels, device);
  customVingetteGPU(*ret, *this, *vingetteG);    
  return ret;
}


ImageImpl* ImageImplGPU::widthVingette(const ImageImpl* vingette, const float treshold) const {
  assert(vingette->isGPU());
  const ImageImplGPU* vingetteG = (ImageImplGPU*) vingette;
  assert(vingetteG->device == device);
  cuda_err_print(cudaSetDevice(device));
  ImageImplGPU *ret = new ImageImplGPU(nRows, nCols, nChannels, device);
  widthVingetteGPU(*ret, *this, *vingetteG, treshold);    
  return ret;
}


void ImageImplGPU::rect_vignetting(size_t filter_size) {
  rect_vignettingGPU(*this, filter_size);
}


ImageImplGPU* ImageImplGPU::operator| (const ImageImplGPU& other) const {
  assert(other.device == device); 
  cuda_err_print(cudaSetDevice(device));
  ImageImplGPU *ret = new ImageImplGPU(nRows, nCols, nChannels, device);
  pixelOrGPU(*ret, *this, other);    
  return ret;
}


ImageImplGPU* ImageImplGPU::operator& (const ImageImplGPU& other) const {
  assert(other.device == device); 
  cuda_err_print(cudaSetDevice(device));
  ImageImplGPU *ret = new ImageImplGPU(nRows, nCols, nChannels, device);
  pixelAndGPU(*ret, *this, other);    
  return ret;
}

ImageImpl* ImageImplGPU::threshold(const float val) {
  cuda_err_print(cudaSetDevice(device));  
  ImageImplGPU* ret = new ImageImplGPU(nRows, nCols, nChannels, device);
  thresholdGPU(*ret, *this, val);
  return ret;
  }


ImageImpl* ImageImplGPU::erode() {
  cuda_err_print(cudaSetDevice(device));  
  ImageImplGPU* ret = new ImageImplGPU(nRows, nCols, nChannels, device);
  erodeGPU(*ret, *this);
  return ret;
}

ImageImpl* ImageImplGPU::fill (const float value) const {
  ImageImplGPU* ret = new ImageImplGPU(nRows, nCols, nChannels, device);
  cuda_err_print(cudaSetDevice(device)); 
  setGPU(*ret, {value, value});
  return ret;
}
