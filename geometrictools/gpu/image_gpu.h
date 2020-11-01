#include "utils/image_cpu.h"
#include "gpu/interval.h"

using namespace std;

#pragma once

class ImageImplGPU : public ImageImpl {
public:
  IntervalGPU *data = 0;
  const size_t device;
  ImageImplGPU(size_t nRows, size_t nCols, size_t nChannels, size_t device);
  ImageImplGPU(const ImageImpl &other, const size_t device);
  ImageImpl& operator = (const ImageImpl &other);
  ~ImageImplGPU();

  void copyFrom(const ImageImpl &other);
  void copyTo(Interval* other) const;
  
  virtual bool isGPU() const {return true;};
  virtual ImageImpl* resize(const size_t new_nRows, const size_t new_nCols, const bool roundToInt) const;
  virtual ImageImpl* resize(const size_t new_nRows, const size_t new_nCols, const bool roundToInt, const size_t size) const;
  ImageImplGPU* operator + (const ImageImplGPU& other) const;
  ImageImplGPU* operator | (const ImageImplGPU& other) const;
  ImageImplGPU* operator & (const ImageImplGPU& other) const;  
  ImageImplGPU* operator - (const ImageImplGPU& other) const;
  virtual void saveBMP(string fn, bool inf) const;
  virtual float l2norm() const;
  virtual std::tuple<float, float, float> l2norm_channelwise() const;
  virtual void zero();
  
  virtual void roundToInt();
  virtual void clip(const float min_val, const float max_val);
  
  virtual ImageImpl* filter_vignetting(const float* filter, size_t filter_size, float radiusDecrease) const;
  virtual float* getFilter(float sigma, size_t filter_size) const;
  virtual void delFilter(float *f) const;
  //virtual std::vector<ImageImpl*> split_color_channels() const;
  virtual ImageImpl* center_crop(const size_t size) const;  

  virtual ImageImpl* rotate(const Interval& params, const bool roundToInt) const;
  ImageImpl* rotate(const vector<Interval> params, const bool singleToMany, const bool roundToInt) const;

  virtual vector<float> l2normC(size_t c) const;
  virtual vector<float> l2diffC(const ImageImplGPU& other, size_t c) const;


  ImageImpl* translate(const Interval& dx,
                       const Interval& dy,
                                     const bool roundToInt) const;
  
  ImageImpl* translate(const vector<Interval> dx,
                       const vector<Interval> dy,
                       const bool roundToInt) const;
  
  vector<float> filterVignettingL2diffC(const ImageImplGPU& other,
                                        const ImageImplGPU* vingette,
                                        const float* filter, size_t filter_size,
                                        float radiusDecrease, const size_t c);
  


  virtual bool inverseRot(ImageImpl** out_ret,
                          ImageImpl** out_vingette,
                          const Interval gamma,
                          const size_t refinements,
                          const bool addIntegerError,
                          const bool toIntegerValues,
                          const bool computeVingette) const;
  virtual bool inverseTranslation(ImageImpl** out_ret,
                                  ImageImpl** out_vingette,
                                  const Interval dx,
                                  const Interval dy,
                                  const size_t refinements,
                                  const bool addIntegerError,
                                  const bool toIntegerValues,
                                  const bool computeVingette) const;

  virtual void rect_vignetting(size_t filter_size);

  virtual ImageImpl* customVingette(const ImageImpl* vingette) const;
  virtual ImageImpl* widthVingette(const ImageImpl* vingette, const float treshold) const;

  virtual ImageImpl* threshold(const float val);
  virtual ImageImpl* erode();
  virtual ImageImpl* fill (const float value) const;
};
