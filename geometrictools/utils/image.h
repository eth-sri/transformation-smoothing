#include "utils/image_cpu.h"
#include "gpu/image_gpu.h"
#include "domains/interval.h"
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <complex>
#include <iostream>
#include <tuple>
#include <algorithm>
#include "utils/constants.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <memory>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <random>

using namespace std;

#pragma once

class Image {

  // private:

public:
  shared_ptr<ImageImpl> img;

  /*** constuctors and friends ***/

  //empty
  Image() {};

  Image(ImageImpl* img) {
    this->img.reset(img);
  };  
    
  //from numpy
  Image(PyArrayObject* np) {
    img.reset(new ImageImpl(np));
  };

  //copy
  Image(const Image &other){
    img = other.img;
  };


  Image copy() const {
    assert(img);
    if (isGPU()) {
      ImageImplGPU* thisImg = (ImageImplGPU *)img.get();
      return Image(new ImageImplGPU(*thisImg, thisImg->device));
  } else {
      return Image(new ImageImpl(*img.get()));    
    }
  };
  
  // from string
  Image(size_t nRows, size_t nCols, size_t nChannels, std::string line) {
    img.reset(new ImageImpl(nRows, nCols, nChannels, line));    
  };

  // empty with fixed size
  Image(size_t nRows, size_t nCols, size_t nChannels) {
    img.reset(new ImageImpl(nRows, nCols, nChannels));    
  };
  
  // assignment

  Image& operator= (const Image &other) {
    img = other.img;
    return *this;
  };
  

  bool isGPU() const {
    if (img) return img->isGPU();
    return false;
  };


  /*** other ***/  
  Image operator + (const Image& other) const {
    assert(img);
    assert(other.img);
    assert(other.isGPU() == isGPU());
    assert(other.isGPU() == isGPU());
    if (isGPU()) {
      return Image(*(ImageImplGPU*)img.get() + *(ImageImplGPU*)other.img.get());
    } else {
      return Image(*img + *other.img);
    }
  };

  Image operator | (const Image& other) const {
    assert(img);
    assert(other.img);
    assert(other.isGPU() == isGPU());
    if (isGPU()) {
      return Image(*(ImageImplGPU*)img.get() | *(ImageImplGPU*)other.img.get());
    } else {
      return Image(*img | *other.img);
    }
  };

  Image operator & (const Image& other) const {
    assert(img);
    assert(other.img);
    assert(other.isGPU() == isGPU());
    if (isGPU()) {
      return Image(*(ImageImplGPU*)img.get() & *(ImageImplGPU*)other.img.get());
    } else {
      return Image(*img & *other.img);
    }
  };
  
  Image operator - (const Image& other) const {
    assert(img);
    assert(other.img);
    assert(other.isGPU() == isGPU());
    if (isGPU()) {
      return Image(*(ImageImplGPU*)img.get() - *(ImageImplGPU*)other.img.get());
    } else {
      return Image(*img - *other.img);
    }
  };
  
  void saveBMP(string fn, bool inf) const {
    assert(img);
    (*this).to_cpu().img->saveBMP(fn, inf);
  };
  
  float l2norm() const {
    assert(img);    
    return img->l2norm();
  };

  vector<float> l2normC(size_t c) const {
    assert(img);
    assert(c <= nChannels());
    if (c == nChannels()) {
      return {img->l2norm()};
    } else if (isGPU()) {
      return ((ImageImplGPU*)img.get())->l2normC(c);
    }
    else if (!isGPU() && c == 1) {
      std::vector<Image> chans =  split_color_channels();
      std::vector<float> out;
      for(auto i : chans) {
        out.push_back(i.l2norm());
      }
      return out;
    } else {
      assert(false);
    }
    return {};
  };

  vector<float> l2diffC(const Image& other, size_t c) const {
    assert(img);
    assert(other.img);
    assert(nChannels() == other.nChannels());
    if(isGPU() && other.isGPU()) {
      return ((ImageImplGPU*)img.get())->l2diffC(*((ImageImplGPU*)other.img.get()), c);
    } else if (!isGPU() && !other.isGPU()) {
      return (other - *this).l2normC(c);
    } else {
      assert(false);
    }
    return {};
  }
  
  float* getFilter(float sigma, size_t filter_size) const {
    assert(img);    
    return img->getFilter(sigma, filter_size);
  }

  void delFilter(float* f) const {
    assert(img);    
    return img->delFilter(f);
  }
  
  Image filter_vignetting(const float* filter, size_t filter_size, float radiusDecrease) const {
    assert(img);    
    return Image(img->filter_vignetting(filter, filter_size, radiusDecrease));
  };

  Image filter_vignetting(float sigma, size_t filter_size, float radiusDecrease) const {
    assert(img);
    float *filter = 0;
    if (sigma > 0)
      filter = getFilter(sigma, filter_size);
    
    Image out(img->filter_vignetting(filter, filter_size, radiusDecrease));
    delFilter(filter);
    return out;
  };
    
  void roundToInt() {
    assert(img);
    img->roundToInt();
  };

  void clip(const float min_val, const float max_val) {
    assert(img);
    img->clip(min_val, max_val);
  }
  
  std::vector<Image> split_color_channels() const {
    assert(img);    
    std::vector<ImageImpl*> imgs = img->split_color_channels();
    std::vector<Image> out;
    for (ImageImpl *img : imgs) {
      out.push_back(Image(img));
    }
    return out;
  };
  
  Image resize(const size_t new_nRows, const size_t new_nCols, const bool roundToInt) const {
    assert(img);    
    return Image(img->resize(new_nRows, new_nCols, roundToInt));
  };


  Image resize(const size_t size, const bool roundToInt) const {
    assert(img);    
    const size_t w = img->nCols;
    const size_t h = img->nRows;
      
    if ((w <= h && w == size) || (h <= w && h == size)) {
      return *this;
    } else if (w < h) {
      const size_t ow = size;
      const size_t oh = int(size * 1.0 * h / w);
      return resize(oh, ow, roundToInt);
    } else {
      const size_t oh = size;
      const size_t ow = int(size * 1.0 * w / h);
      return resize(oh, ow, roundToInt);
    }
  };

  
  Image resize(const size_t new_nRows, const size_t new_nCols, const bool roundToInt, const size_t centerCropSize) const {
    assert(img);
    if ((centerCropSize == 0)  && !(new_nRows == 0 || new_nCols == 0))
      return Image(img->resize(new_nRows, new_nCols, roundToInt));
    else if ((new_nRows == 0 || new_nCols == 0) && centerCropSize != 0)
      return Image(img->center_crop(centerCropSize));
    else
      return Image(img->resize(new_nRows, new_nCols, roundToInt, centerCropSize));
  };


  Image resize(const size_t size, const bool roundToInt, const size_t centerCropSize) const {
    assert(img);
    const size_t w = img->nCols;
    const size_t h = img->nRows;

    if ((centerCropSize == 0)  && size != 0)
      return resize(size, roundToInt);
    else if (((size == 0) || (w <= h && w == size) || (h <= w && h == size) ) && centerCropSize != 0)
      return Image(img->center_crop(centerCropSize));
    else {
      size_t ow = 0;
      size_t oh = 0;
      if (w < h) {
        ow = size;
        oh = int(size * 1.0 * h / w);
      } else {
        oh = size;
        ow = int(size * 1.0 * w / h);
      }
      return Image(img->resize(oh, ow, roundToInt, centerCropSize));    
    }   
  };
  
  vector<float> filterVignettingL2diffC(const Image& other, const float* filter, const size_t filter_size, const float radiusDecrease,
                                        const size_t c) {
    assert(img);
    assert(other.img);
    assert(nChannels() == other.nChannels());
    assert(c <= nChannels());

    if (isGPU() && other.isGPU()) {    
      vector<float> out = ((ImageImplGPU*)img.get())->filterVignettingL2diffC(*((ImageImplGPU*)other.img.get()),
                                                                              0, filter, filter_size,
                                                                              radiusDecrease, c);
      return out;
    }
    else if (!isGPU() && !other.isGPU()) {
      Image a = (img->filter_vignetting(filter, filter_size, radiusDecrease));
      Image b = (other.img->filter_vignetting(filter, filter_size, radiusDecrease));
      return a.l2diffC(b, c);      
    }
    else assert(false);
  };


  Image customVingette(const Image& vingette) {
    assert(img);
    assert(vingette.img);
    assert(nChannels() == vingette.nChannels());
    return Image(img->customVingette(vingette.img.get()));
  }

  Image widthVingette(const Image& vingette, const float treshold) {
    assert(img);
    assert(vingette.img);
    assert(nChannels() == vingette.nChannels());
    return Image(img->widthVingette(vingette.img.get(), treshold));
  }

  

  vector<float> filterVignettingL2diffCWithCustomVingette(const Image& other,
                                                          const Image& vingette,
                                                          const float* filter,
                                                          const size_t filter_size,
                                                          const float radiusDecrease,
                                                          const size_t c) {
    assert(img);
    assert(other.img);
    assert(vingette.img);
    assert(nChannels() == other.nChannels());
    assert(nChannels() == vingette.nChannels());
    assert(c <= nChannels());

    if (isGPU() && other.isGPU()) {    
      vector<float> out = ((ImageImplGPU*)img.get())->filterVignettingL2diffC(*((ImageImplGPU*)other.img.get()),
                                                                              ((ImageImplGPU*)vingette.img.get()),
                                                                              filter, filter_size, radiusDecrease,
                                                                                                c);
      return out;
    }
    else if (!isGPU() && !other.isGPU()) {
      Image a = (img->filter_vignetting(filter, filter_size, radiusDecrease));
      Image b = (other.img->filter_vignetting(filter, filter_size, radiusDecrease));
      a = a.customVingette(vingette);
      b = b.customVingette(vingette);
      return a.l2diffC(b, c);      
    }
    else assert(false);
  };
  
  Image center_crop(const size_t size) const {
    assert(img);    
    return Image(img->center_crop(size));
  };

  size_t nChannels() const {
    assert(img);
    return img->nChannels;
  };

  size_t nCols() const {
    assert(img);
    return img->nCols;
  };

  size_t nRows() const {
    assert(img);
    return img->nRows;
  };
  
  
  Image channels(size_t start, size_t length) const {
    assert(img);
    assert(start == 0);
    if(length == nChannels()) return *this;
    assert(!isGPU());    
    assert(length <= nChannels());
    ImageImpl* im = new ImageImpl(nRows(), nCols(), length);

    for (size_t k = 0; k < length; ++k) {
      for (size_t i = 0; i < nRows(); ++i) {
        for (size_t j = 0; j < nCols(); ++j) {
          size_t m = img->rcc_to_idx(i, j, start + k);
          size_t n = im->rcc_to_idx(i, j, k);
          im->a[n] = img->a[m];
        }
      }
    }

    return Image(im);
  };

  Image inverseRotation(Interval gamma, bool& isEmpty, size_t refinements, bool addIntegerError, bool toIntegerValues) const {
    assert(img);
    ImageImpl* ret = 0;
    isEmpty = img->inverseRot(&ret,
                              0,
                              gamma,                              
                              refinements,
                              addIntegerError,
                              toIntegerValues,
                              false);
    return Image(ret);
  };

  Image inverseTranslation(Interval dx, Interval dy, bool& isEmpty, size_t refinements, bool addIntegerError, bool toIntegerValues) const {
    assert(img);

    ImageImpl* ret = 0;
    isEmpty = img->inverseTranslation(&ret,
                                      0,
                                      dx,
                                      dy,
                                      refinements,
                                      addIntegerError,
                                      toIntegerValues,
                                      false);
    return Image(ret);
  };

  tuple<Image, Image> inverseRotationWithVingette(Interval gamma, bool& isEmpty, size_t refinements, bool addIntegerError, bool toIntegerValues) const {
    assert(img);

    ImageImpl* ret = 0;
    ImageImpl* vingette = 0;
    isEmpty = img->inverseRot(&ret,
                              &vingette,
                              gamma,                              
                              refinements,
                              addIntegerError,
                              toIntegerValues,
                              true);
    return {Image(ret), Image(vingette)};
  };

  tuple<Image, Image> inverseTranslationWithVingette(Interval dx, Interval dy, bool& isEmpty, size_t refinements, bool addIntegerError, bool toIntegerValues) const {
    assert(img);

    ImageImpl* ret = 0;
    ImageImpl* vingette = 0;
    isEmpty = img->inverseTranslation(&ret,
                                      &vingette,
                                      dx,
                                      dy,
                                      refinements,
                                      addIntegerError,
                                      toIntegerValues,
                                      true);
    return {Image(ret), Image(vingette)};
  };

  
  Image rotate(const vector<Interval>& params, const bool roundToInt) const {
    assert(img);
    if (isGPU()) {
      return Image(((ImageImplGPU*)img.get())->rotate(params, true, roundToInt));      
    } else {
      assert(params.size() == 1);
      return Image(img->rotate(params[0], roundToInt));
    }
  };

  Image rotate(const vector<float>& params, const bool roundToInt) const {
    assert(img);
    vector<Interval> paramsI;
    for(float p : params) {
      paramsI.push_back({p, p});
    }
    return rotate(paramsI, roundToInt);
  };
  
  Image rotate(const Interval& params, const bool roundToInt) const {
    assert(img);
    return Image(img->rotate(params, roundToInt));
  };
  
  Image rotate(const float& params, const bool roundToInt) const {
    return rotate(Interval(params, params), roundToInt);
  };

  Image translate(const Interval& dx, const Interval& dy, const bool roundToInt) const {
    assert(img);
    return Image(img->translate(dx, dy, roundToInt));
  };

  Image translate(const float dx, const float dy, const bool roundToInt) const {
    assert(img);
    return Image(img->translate(Interval(dx, dx), Interval(dy, dy), roundToInt));
  };


  Image rect_vignetting(size_t filter_size) {
    assert(img);
    img->rect_vignetting(filter_size);
    return *this;
  };  
  
  
  Image translate(const vector<Interval>& dx, const vector<Interval>& dy, const bool roundToInt) const {
    assert(dx.size() == dy.size());
    assert(img);
    if (isGPU()) {
      return Image(((ImageImplGPU*)img.get())->translate(dx, dy,
                                                         roundToInt));      
    } else {
      assert(dx.size() == 1);
      return Image(img->translate(dx[0], dy[0], roundToInt));
    }
  }; 

  Image translate(const vector<float>& dx, const vector<float>& dy, const bool roundToInt) const {
    assert(dx.size() == dy.size());
    vector<Interval> dxI;
    vector<Interval> dyI;
    for(size_t k = 0; k < dx.size(); ++k) {
      dxI.push_back({dx[k], dx[k]});
      dyI.push_back({dy[k], dy[k]});
    }
    return translate(dxI, dyI, roundToInt);
  }; 
  
  int device() const {
    if (isGPU()) {
      return ((ImageImplGPU *)img.get())->device;
    }
    return -1;
  }
  
  Image to_gpu(const size_t device) const {
    assert(img);
    //cout << "to gpu" << endl;
    if (isGPU()) {
      if (((ImageImplGPU *)img.get())->device == device) {
        return *this; 
      } else {
        ImageImplGPU *img_gpu = new ImageImplGPU(*img, device);
        return Image(img_gpu);
        //return to_cpu().to_gpu(device);
      }
    }
    ImageImplGPU *img_gpu = new ImageImplGPU(*img, device);
    return Image(img_gpu);
  };

  Image to_cpu() const {
    if (!isGPU()) return *this;
    assert(img);
    //cout << "to cpu" << endl;
    ImageImplGPU *img_gpu = (ImageImplGPU *) img.get();
    ImageImpl *img_cpu = new ImageImpl(img->nRows, img->nCols, img->nChannels);
    img_gpu->copyTo(img_cpu->a);
    return Image(img_cpu);
  };

  Image append(const Image& other) {
    assert(img);
    assert(!isGPU() && !other.isGPU());
    assert(nRows() == other.nRows() && nCols() == other.nCols());
    assert(img->a != 0);
    assert(other.img->a != 0);

    
    Image out(nRows(), nCols(), nChannels() + other.nChannels());
    size_t s1 = nRows() * nCols() * nChannels();
    for (size_t i = 0; i < nChannels(); ++i) {
      for (size_t r = 0; r < nRows(); ++r) {
        for (size_t c = 0; c < nCols(); ++c) {
          const size_t m = img->rcc_to_idx(r, c, i);
          const size_t n = out.img->rcc_to_idx(r, c, i);
          out.img->a[n] = img->a[m];
        }
      }
    }

    for (size_t i = 0; i < other.nChannels(); ++i) {
      for (size_t r = 0; r < nRows(); ++r) {
        for (size_t c = 0; c < nCols(); ++c) {
          const size_t m = other.img->rcc_to_idx(r, c, i);
          const size_t n = out.img->rcc_to_idx(r, c, i + nChannels());
          out.img->a[n] = other.img->a[m];
        }
      }
    }    
    return out;
  };

  Image fill(const float value) {
    assert(img);
    return Image(img->fill(value));
  };
  
  
  bool contains(const Image& other) {
    assert(img);
    assert(!isGPU() && !other.isGPU());
    bool check = nRows() == other.nRows() && nCols() == other.nCols() && nChannels() == other.nChannels();
    if (check) {
      for (size_t i = 0; i < nChannels(); ++i) {
        for (size_t r = 0; r < nRows(); ++r) {
          for (size_t c = 0; c < nCols(); ++c) {
            const size_t m = img->rcc_to_idx(r, c, i);
            check = check && img->a[m].contains(other.img->a[m]);
          }
        }
      }
    }
    return check;
  };

  bool equals(const Image& other, const float eps) {
    assert(img);
    assert(!isGPU() && !other.isGPU());
    bool check = nRows() == other.nRows() && nCols() == other.nCols() && nChannels() == other.nChannels();
    if (check) {
      for (size_t i = 0; i < nChannels(); ++i) {
        for (size_t r = 0; r < nRows(); ++r) {
          for (size_t c = 0; c < nCols(); ++c) {
            const size_t m = img->rcc_to_idx(r, c, i);
            Interval a = img->a[m];
            Interval b = other.img->a[m];
            bool checkPixel = ((a.is_empty() && b.is_empty()) ||
                               (a.inf == b.inf && a.sup == b.sup) || //needed for infinity cases
                               (abs(a.inf - b.inf) <= eps &&
                                abs(a.sup - b.sup) <= eps));
            check = check && checkPixel;             
          }
        }
      }
    }
    return check;
  };  

  void randomize(std::default_random_engine& gen, const float p_empty) {
    assert(!isGPU());
    std::uniform_real_distribution<float> distribution(0,1);
    for (size_t i = 0; i < nChannels(); ++i) {
      for (size_t r = 0; r < nRows(); ++r) {
        for (size_t c = 0; c < nCols(); ++c) {
          const size_t m = img->rcc_to_idx(r, c, i);          
          if (distribution(gen) <= p_empty) {
            img->a[m] = Interval();
          } else {
            float lower = distribution(gen);
            float upper = 0;
            while(upper < lower) {
              upper = distribution(gen);
            }
            img->a[m] = Interval(lower, upper);        
          }
        }
      }    
    }
  };

  void randomize(std::default_random_engine& gen ) {
    randomize(gen, 0);
  };
  
  Image threshold(const float val) {
    assert(img);
    return Image(img->threshold(val));
  };

  Image erode() {
    assert(img);
    return Image(img->erode());
  };

  
  friend ostream& operator<<(ostream& os, const Image& img)
  {
    return (cout << *(img.to_cpu().img));
  };
  
};






