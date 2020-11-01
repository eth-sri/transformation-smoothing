#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "utils/utilities.h"
#include "domains/interval.h"
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <complex>
#include <iostream>
#include <tuple>
#include <functional>
#include <algorithm>
#include "utils/constants.h"
#include <Python.h>
#include <numpy/arrayobject.h>

using namespace std;

#pragma once


enum Corner {
  lower_right=0,
  upper_right=1,
  lower_left=2,
  upper_left=3
};


const float pixelVals[256] = {0/255.0f, 1/255.0f, 2/255.0f, 3/255.0f, 4/255.0f, 5/255.0f, 6/255.0f, 7/255.0f, 8/255.0f, 9/255.0f, 10/255.0f, 11/255.0f, 12/255.0f, 13/255.0f, 14/255.0f, 15/255.0f, 16/255.0f, 17/255.0f, 18/255.0f, 19/255.0f, 20/255.0f, 21/255.0f, 22/255.0f, 23/255.0f, 24/255.0f, 25/255.0f, 26/255.0f, 27/255.0f, 28/255.0f, 29/255.0f, 30/255.0f, 31/255.0f, 32/255.0f, 33/255.0f, 34/255.0f, 35/255.0f, 36/255.0f, 37/255.0f, 38/255.0f, 39/255.0f, 40/255.0f, 41/255.0f, 42/255.0f, 43/255.0f, 44/255.0f, 45/255.0f, 46/255.0f, 47/255.0f, 48/255.0f, 49/255.0f, 50/255.0f, 51/255.0f, 52/255.0f, 53/255.0f, 54/255.0f, 55/255.0f, 56/255.0f, 57/255.0f, 58/255.0f, 59/255.0f, 60/255.0f, 61/255.0f, 62/255.0f, 63/255.0f, 64/255.0f, 65/255.0f, 66/255.0f, 67/255.0f, 68/255.0f, 69/255.0f, 70/255.0f, 71/255.0f, 72/255.0f, 73/255.0f, 74/255.0f, 75/255.0f, 76/255.0f, 77/255.0f, 78/255.0f, 79/255.0f, 80/255.0f, 81/255.0f, 82/255.0f, 83/255.0f, 84/255.0f, 85/255.0f, 86/255.0f, 87/255.0f, 88/255.0f, 89/255.0f, 90/255.0f, 91/255.0f, 92/255.0f, 93/255.0f, 94/255.0f, 95/255.0f, 96/255.0f, 97/255.0f, 98/255.0f, 99/255.0f, 100/255.0f, 101/255.0f, 102/255.0f, 103/255.0f, 104/255.0f, 105/255.0f, 106/255.0f, 107/255.0f, 108/255.0f, 109/255.0f, 110/255.0f, 111/255.0f, 112/255.0f, 113/255.0f, 114/255.0f, 115/255.0f, 116/255.0f, 117/255.0f, 118/255.0f, 119/255.0f, 120/255.0f, 121/255.0f, 122/255.0f, 123/255.0f, 124/255.0f, 125/255.0f, 126/255.0f, 127/255.0f, 128/255.0f, 129/255.0f, 130/255.0f, 131/255.0f, 132/255.0f, 133/255.0f, 134/255.0f, 135/255.0f, 136/255.0f, 137/255.0f, 138/255.0f, 139/255.0f, 140/255.0f, 141/255.0f, 142/255.0f, 143/255.0f, 144/255.0f, 145/255.0f, 146/255.0f, 147/255.0f, 148/255.0f, 149/255.0f, 150/255.0f, 151/255.0f, 152/255.0f, 153/255.0f, 154/255.0f, 155/255.0f, 156/255.0f, 157/255.0f, 158/255.0f, 159/255.0f, 160/255.0f, 161/255.0f, 162/255.0f, 163/255.0f, 164/255.0f, 165/255.0f, 166/255.0f, 167/255.0f, 168/255.0f, 169/255.0f, 170/255.0f, 171/255.0f, 172/255.0f, 173/255.0f, 174/255.0f, 175/255.0f, 176/255.0f, 177/255.0f, 178/255.0f, 179/255.0f, 180/255.0f, 181/255.0f, 182/255.0f, 183/255.0f, 184/255.0f, 185/255.0f, 186/255.0f, 187/255.0f, 188/255.0f, 189/255.0f, 190/255.0f, 191/255.0f, 192/255.0f, 193/255.0f, 194/255.0f, 195/255.0f, 196/255.0f, 197/255.0f, 198/255.0f, 199/255.0f, 200/255.0f, 201/255.0f, 202/255.0f, 203/255.0f, 204/255.0f, 205/255.0f, 206/255.0f, 207/255.0f, 208/255.0f, 209/255.0f, 210/255.0f, 211/255.0f, 212/255.0f, 213/255.0f, 214/255.0f, 215/255.0f, 216/255.0f, 217/255.0f, 218/255.0f, 219/255.0f, 220/255.0f, 221/255.0f, 222/255.0f, 223/255.0f, 224/255.0f, 225/255.0f, 226/255.0f, 227/255.0f, 228/255.0f, 229/255.0f, 230/255.0f, 231/255.0f, 232/255.0f, 233/255.0f, 234/255.0f, 235/255.0f, 236/255.0f, 237/255.0f, 238/255.0f, 239/255.0f, 240/255.0f, 241/255.0f, 242/255.0f, 243/255.0f, 244/255.0f, 245/255.0f, 246/255.0f, 247/255.0f, 248/255.0f, 249/255.0f, 250/255.0f, 251/255.0f, 252/255.0f, 253/255.0f, 254/255.0f, 255/255.0f};

class ImageImpl {
public:
  Interval *a = 0;
  size_t nRows, nCols, nChannels;

  /*** constuctors and friends ***/

  //empty
  ImageImpl();

  // from torch
  // ImageImpl(const torch::Tensor &img)
  
  //from numpy
  ImageImpl(PyArrayObject* np);

  //copy
  ImageImpl(const ImageImpl &other);
  
  //empty fixed size
  ImageImpl(size_t nRows, size_t nCols, size_t nChannels);
  
  // from string
  ImageImpl(size_t nRows, size_t nCols, size_t nChannels, std::string line);

  // assignment
  virtual ImageImpl& operator = (const ImageImpl &other);
  
  // destructor
  virtual ~ImageImpl();
  

  /*** other ***/
  virtual bool isGPU() const {return false;};
  size_t rcc_to_idx(size_t r, size_t c, size_t ch) const {
    return ch * this->nCols * this->nRows
      + r  * this->nCols
      + c;
  };
  ImageImpl* operator + (const ImageImpl& other) const;
  ImageImpl* operator | (const ImageImpl& other) const;
  ImageImpl* operator & (const ImageImpl& other) const;
  ImageImpl* operator - (const ImageImpl& other) const;
  virtual void saveBMP(string fn, bool inf) const; 
  virtual Interval valueAt(int x, int y, size_t i) const;
  virtual Interval valueAt(int x, int y, size_t i, Interval default_value) const;
  virtual Coordinate<float> getCoordinate(float r, float c, size_t i) const;
  virtual float l2norm() const;
  virtual std::tuple<float, float, float> l2norm_channelwise() const;
  virtual ImageImpl* filter_vignetting(const float* filter, size_t filter_size, float radiusDecrease) const;
  virtual void zero();
  virtual void roundToInt();
  virtual void clip(const float min_val, const float max_val);
  virtual std::vector<ImageImpl*> split_color_channels()const ;
  virtual ImageImpl* resize(const size_t new_nRows, const size_t new_nCols, const bool roundToInt) const;
  virtual ImageImpl* resize(const size_t new_nRows, const size_t new_nCols, const bool roundToInt, const size_t size) const;
  virtual ImageImpl* center_crop(const size_t size) const;  

  
  virtual float* getFilter(float sigma, size_t filter_size) const;
  virtual void delFilter(float *f) const;

  vector<float> bilinear_coefficients(int x_iter, int y_iter, float x, float y) const;
  std::tuple<Interval, Interval, Interval, Interval> inverse_interval(Interval p, int x_iter, int y_iter, Interval x_box, Interval y_box) const;

  virtual void rect_vignetting(size_t filter_size);
  virtual ImageImpl* customVingette(const ImageImpl* vingette) const;
  virtual ImageImpl* widthVingette(const ImageImpl* vingette, const float treshold) const;
  
  ImageImpl* transform(std::function<Coordinate<Interval>(const Coordinate<float>)> T,
                       const bool roundToInt) const;
  
  virtual ImageImpl* rotate(const Interval& params, const bool roundToInt) const;
  virtual ImageImpl* translate(const Interval& dx, const Interval& dy, const bool roundToInt) const;
 


  virtual ImageImpl* fill (const float value) const;
  
  
  std::tuple<int, int> getRc(float x, float y) const;

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
  

  friend ostream& operator<<(ostream& os, const ImageImpl& dt);

  bool invert_pixel_on_bounding_box(const size_t nRow,
                                    const size_t nCol,
                                    ImageImpl *ret,
                                    const bool refine,
                                    const ImageImpl *constraints,
                                    const std::function<Coordinate<Interval>(Coordinate<Interval>)> T,
                                    const std::function<Coordinate<Interval>(Coordinate<Interval>)> invT,
                                    const bool addIntegerError,
                                    const bool toIntegerValues,
                                    const bool computeVingette,
                                    ImageImpl *vingette) const;


  bool inverse(ImageImpl **out,
               ImageImpl **vingette,
               const std::function<Coordinate<Interval>(Coordinate<Interval>)> T,
               const std::function<Coordinate<Interval>(Coordinate<Interval>)> invT,
               const bool addIntegerError,
               const bool toIntegerValues,
               const size_t nrRefinements,
               const bool computeVingette) const;
  
  void invert_pixel(const Corner center,
                    const Coordinate<float>& coord,
                    const Coordinate<Interval>& coord_iterT,
                    Interval* out,
                    const Interval* pixel_values,
                    const ImageImpl* constraints,
                    const bool refine,
                    const bool debug) const;

  
  virtual ImageImpl* threshold(const float val);
  virtual ImageImpl* erode();
};
