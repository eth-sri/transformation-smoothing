#include "image.h"
#include "image_cpu.h"
#include "transforms/interpolation.h"
#include "utils/utilities.h"

using namespace std;

Interval roundIntervalToInt(const Interval& i) {
  Interval in = i.meet({0, 1});
  float inf = in.inf * 255.0f;  
  int inf_lower = floorf(inf);

  if (inf_lower == roundf(inf)) {
    inf = pixelVals[inf_lower];
  } else {
    inf = pixelVals[min(255, inf_lower + 1)];
  }

  float sup = in.sup * 255.0f;  
  int sup_lower = floorf(sup);
  if (sup_lower == roundf(sup)) {
    sup = pixelVals[sup_lower];
  } else {
    sup = pixelVals[min(255, sup_lower + 1)];
  }

  assert(inf <= sup);  
  return {inf, sup};
}

/*** constuctors and friends ***/

//empty
ImageImpl::ImageImpl() {};

// from torch
// ImageImpl(const torch::Tensor &img) {
//   assert(img.dim() == 3);
//   this->nRows = img.size(1);
//   this->nCols = img.size(2);
//   this->nChannels = img.size(0);    
//   const size_t s = nRows * nCols * nChannels;
//   auto acc = img.accessor<float, 3>();
//   this->a = new Interval[s];

//   float _min = 0;
//   float _max = 0;
//   for(size_t i = 0; i < nRows; ++i)
//       for(size_t j = 0; j < nCols; ++j)
//         for(size_t k = 0; k < nChannels; ++k)
//           {
//             const size_t m = rcc_to_idx(i, j, k);
//             const float v = acc[k][i][j];
//             a[m] = {v, v};
//             _min = min(_min, v);
//             _max = max(_max, v);
//           }
//   cout << _min << " " << _max << endl;
// }

//from numpy
ImageImpl::ImageImpl(PyArrayObject* np) {
  int dims = PyArray_NDIM(np);
  assert(dims == 2 || dims == 3);
  this->nRows = PyArray_DIM(np, 0);
  this->nCols = PyArray_DIM(np, 1);
  if (dims == 2) {
    this->nChannels = 1;
  } else {
    this->nChannels = PyArray_DIM(np, 2);
  }
  
  int image_typ = PyArray_TYPE(np);
  if (image_typ == NPY_UINT8) {
    const size_t s = nRows * nCols * nChannels;
    uint8_t* acc = (uint8_t*)PyArray_DATA(np);
    this->a = new Interval[s];

    for(size_t i = 0; i < nRows; ++i)
      for(size_t j = 0; j < nCols; ++j)
        for(size_t k = 0; k < nChannels; ++k)
          {
            const uint8_t v = acc[i * nCols * nChannels + j * nChannels + k ];
            const float vv = v / 255.0f; 
            const size_t m = rcc_to_idx(i, j, k);
            a[m] = {vv, vv};
          }    
  } else if (image_typ == NPY_FLOAT32) {
    const size_t s = nRows * nCols * nChannels;
    float* acc = (float*)PyArray_DATA(np);
    this->a = new Interval[s];

    for(size_t i = 0; i < nRows; ++i)
      for(size_t j = 0; j < nCols; ++j)
        for(size_t k = 0; k < nChannels; ++k)
          {
            const float v = acc[i * nCols * nChannels + j * nChannels + k ];
            assert(v >= 0.0f);
            assert(v <= 255.0f);
            const size_t m = rcc_to_idx(i, j, k);
            a[m] = {v, v};
          }    
  } else {
    assert(false);
  }

} 

//copy
ImageImpl::ImageImpl(const ImageImpl &other) {
  this->nRows = other.nRows;
  this->nCols = other.nCols;
  this->nChannels = other.nChannels;
  const size_t s = nRows * nCols * nChannels;
  this->a = new Interval[s];
  for(size_t i = 0; i < s; ++i)
    a[i] = other.a[i];
}

//empty fixed size
ImageImpl::ImageImpl(size_t nRows, size_t nCols, size_t nChannels) {
  this->nRows = nRows;
  this->nCols = nCols;
  this->nChannels = nChannels;
  this->a = new Interval[nRows * nCols * nChannels];
}

// from string
ImageImpl::ImageImpl(size_t nRows, size_t nCols, size_t nChannels, string line) {
  this->nRows = nRows;
  this->nCols = nCols;
  this->nChannels = nChannels;
  this->a = new Interval[nRows * nCols * nChannels];
  string curr = "";

  // Label of the image is the first digit
  size_t offset = 0;
  while (line[offset] != ',') {
    curr += line[offset++];
  }
  // we currently don't need this
  // this->label = stoi(curr);
  curr = "";

  // ImageImpl
  vector<Interval> its;
  for (size_t i = offset+1; i < line.size(); ++i) {
    if (line[i] == ',') {
      its.push_back({stof(curr), stof(curr)});
      curr = "";
    } else {
      curr += line[i];
    }
  }
  its.push_back({stof(curr), stof(curr)});

  assert((size_t)its.size() == nRows * nCols * nChannels);
  size_t nxt = 0;
  for (size_t r = 0; r < nRows; ++r) {
    for (size_t c = 0; c < nCols; ++c) {
      for (size_t i = 0; i < nChannels; ++i) {
        this->a[rcc_to_idx(r, c, i)] = its[nxt++];
      }
    }
  }
}

// assignment
ImageImpl& ImageImpl::operator = (const ImageImpl &other) 
{
  this->nRows = other.nRows;
  this->nCols = other.nCols;
  this->nChannels = other.nChannels;
  const size_t s = nRows * nCols * nChannels;
  // delete old pointer before changing to prevent memory leak
  if (this->a != 0)
    {
      delete[] this->a;
      this->a = 0;
    }
  this->a = new Interval[s];
  for(size_t i = 0; i < s; ++i)
    a[i] = other.a[i];
  return *this; 
}  

// destructor
ImageImpl::~ImageImpl() {
  if (this->a != 0)
    {
      delete[] this->a;
      this->a = 0;
    }
};


/*** other ***/  

ImageImpl* ImageImpl::operator + (const ImageImpl& other) const {
  assert(nRows==other.nRows && nCols==other.nCols && nChannels==other.nChannels);
  ImageImpl* ret = new ImageImpl(nRows, nCols, nChannels);
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, k);
        ret->a[m] = a[m] + other.a[m];
      }
    }
  }
  return ret;
}

ImageImpl* ImageImpl::operator | (const ImageImpl& other) const {
  assert(nRows==other.nRows && nCols==other.nCols && nChannels==other.nChannels);
  ImageImpl* ret = new ImageImpl(nRows, nCols, nChannels);
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, k);
        if (a[m].inf > 0 || other.a[m].inf > 0)
          ret->a[m] = {1, 1};
        else
          ret->a[m] = {0, 0};
      }
    }
  }
  return ret;
}



ImageImpl* ImageImpl::operator & (const ImageImpl& other) const {
  assert(nRows==other.nRows && nCols==other.nCols && nChannels==other.nChannels);
  ImageImpl* ret = new ImageImpl(nRows, nCols, nChannels);
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, k);
        if (a[m].inf > 0 && other.a[m].inf > 0)
          ret->a[m] = {1, 1};
        else
          ret->a[m] = {0, 0};
      }
    }
  }
  return ret;
}


ImageImpl* ImageImpl::operator - (const ImageImpl& other) const {
  assert(nRows==other.nRows && nCols==other.nCols && nChannels==other.nChannels);
  ImageImpl *ret = new ImageImpl(nRows, nCols, nChannels);
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, k);
        ret->a[m] = a[m] - other.a[m];
      }
    }
  }
  return ret;
}

// Remark: works for even and odd image sizes
Interval ImageImpl::valueAt(int x, int y, size_t i, Interval default_value) const {
  assert(unsigned(x % 2) != nCols % 2 && unsigned(y % 2) != nRows % 2 && "Off grid coordinates"); 
  const int c = (x + (nCols - 1)) / 2;
  const int r = (y + (nRows - 1)) / 2;
  if (r < 0 || r >= signed(nRows) || c < 0 || c >= signed(nCols)) {
    return default_value;
  }
  size_t m = rcc_to_idx(r, c, i);
  assert(a != 0);
  return a[m];
}


Interval ImageImpl::valueAt(int x, int y, size_t i) const {
  return valueAt(x, y, i, {0, 0});
}

// Input:
// 0 <= r <= nRows - 1 
// 0 <= c <= nCols - 1
// Output: coordinates on the odd integer grid
//
// Remark: works for even and odd image sizes
Coordinate<float> ImageImpl::getCoordinate(float r, float c, size_t channel) const {
  return {2 * c - (nCols - 1), 2 * r - (nRows - 1), channel};
}

// Coordinate to rcc inex
std::tuple<int, int> ImageImpl::getRc(float x, float y) const {
  int c = int(int(x + (nCols - 1)) / 2);
  int r = int(int(y + (nRows - 1)) / 2);
  return {r, c};
}


float ImageImpl::l2norm() const {
  Interval sum_squared(0,0);
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, k);
        sum_squared += a[m].square();
      }
    }
  }
  return sqrt(sum_squared.sup);
}


std::tuple<float, float, float> ImageImpl::l2norm_channelwise() const {
  Interval sum_squared_0(0,0), sum_squared_1(0,0), sum_squared_2(0,0);
  for (size_t i = 0; i < nRows; ++i) {
    for (size_t j = 0; j < nCols; ++j) {
      sum_squared_0 += a[rcc_to_idx(i, j, 0)].square();
      sum_squared_1 += a[rcc_to_idx(i, j, 1)].square();
      sum_squared_2 += a[rcc_to_idx(i, j, 2)].square();
    }
  }
  return {sqrt(sum_squared_0.sup), sqrt(sum_squared_1.sup), sqrt(sum_squared_2.sup)};
}

float* ImageImpl::getFilter(float sigma, size_t filter_size) const {
  const size_t c = filter_size / 2;
  // use symmetry to only store 1 quarter of the filter
  float *G = new float[(c+1) * (c+1)];
  getGaussianFilter(G, filter_size, sigma);
  return G;  
}

void ImageImpl::delFilter(float *f) const {
  if (f != 0)
    delete[] f;
}


void ImageImpl::rect_vignetting(size_t filter_size) {
  for (size_t chan = 0; chan < nChannels; ++chan) {  
    for (size_t i = 0; i < min(nRows, filter_size); ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, chan);
        a[m] = {0, 0};
      }
    }

    for (size_t i = max(nRows-filter_size, (size_t)0); i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, chan);
        a[m] = {0, 0};
      }
    }

    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < min(nCols, filter_size); ++j) {
        size_t m = rcc_to_idx(i, j, chan);
        a[m] = {0, 0};
      }
      for (size_t j = max(nRows-filter_size, (size_t)0); j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, chan);
        a[m] = {0, 0};
      }      
    }
  }  
}

ImageImpl* ImageImpl::filter_vignetting(const float* filter, size_t filter_size, float radiusDecrease) const {
  assert(nRows == nCols);
  assert(filter_size % 2 == 1); // odd filter so we can correctly center it
 
  const size_t c = filter_size / 2; 
  const bool do_radius = radiusDecrease >= 0;
  const bool do_filter = filter != 0;
  const float radius = do_radius ? min(nRows, nCols) - 1 - 2 * radiusDecrease : numeric_limits<float>::infinity();
  const float radiusSq = radius*radius;
  ImageImpl *ret = new ImageImpl(nRows, nCols, nChannels);
  for (size_t chan = 0; chan < nChannels; ++chan) {
    for (size_t row = 0; row < nRows; ++row){
      for (size_t col = 0; col < nCols; ++col) {
        const int x =  2 * col - (nCols - 1);
        const int y =  2 * row - (nRows - 1);
        //cout << do_filter << " " << x << " " << y << endl;

        if(x*x + y*y <= radiusSq) {
          if (do_filter) {
            Interval out = {0, 0};
            for (int i = -c; i <= signed(c); ++i)
              for (int j = -c; j <= signed(c); ++j) {
                const int r_ = row + i;
                const int c_ = col + j;
                const int x_ =  2 * c_ - (nCols - 1);
                const int y_ =  2 * r_ - (nRows - 1);
                //cout << "a" << endl << flush;
                //cout << x_*x_ << " " << y_*y_ << " " << radiusSq << endl;
                if (r_ > 0 && r_ < signed(nRows) && c_ > 0 && c_ < signed(nCols)
                    && (x_*x_ + y_*y_ <= radiusSq)
                    ) {
                  out += filter[abs(i) * (c+1) +  abs(j)] * this->a[rcc_to_idx(r_, c_, chan)];
                }
              }          
            ret->a[rcc_to_idx(row, col, chan)] = out;          
          } else {
            ret->a[rcc_to_idx(row, col, chan)] = a[rcc_to_idx(row, col, chan)];
          }          
        } else {
          ret->a[rcc_to_idx(row, col, chan)] = {0,0};
        }
        
        
        
        //cout << rcc_to_idx(row, col, chan) << " " << ret->a[rcc_to_idx(row, col, chan)] << endl; 
      }
    }
  }
  return ret;
}
  
void ImageImpl::saveBMP(string fn, bool inf) const {
  assert(nChannels == 1 || nChannels == 3);

  // based on https://stackoverflow.com/a/18675807
  
  unsigned char file[14] = {
    'B','M', // magic
    0,0,0,0, // size in bytes
    0,0, // app data
    0,0, // app data
    40+14,0,0,0 // start of data offset
  };
  unsigned char info[40] = {
    40,0,0,0, // info hd size
    0,0,0,0, // width
    0,0,0,0, // heigth
    1,0, // number color planes
    24,0, // bits per pixel
    0,0,0,0, // compression is none
    0,0,0,0, // image bits size
    0x13,0x0B,0,0, // horz resoluition in pixel / m
    0x13,0x0B,0,0, // vert resolutions (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
    0,0,0,0, // #colors in pallete
    0,0,0,0, // #important colors
  };

  const size_t w = nCols;
  const size_t h = nRows;
  
  int padSize  = (4-(w*3)%4)%4;
  int sizeData = w*h*3 + h*padSize;
  int sizeAll  = sizeData + sizeof(file) + sizeof(info);

  file[ 2] = (unsigned char)( sizeAll    );
  file[ 3] = (unsigned char)( sizeAll>> 8);
  file[ 4] = (unsigned char)( sizeAll>>16);
  file[ 5] = (unsigned char)( sizeAll>>24);

  info[ 4] = (unsigned char)( w   );
  info[ 5] = (unsigned char)( w>> 8);
  info[ 6] = (unsigned char)( w>>16);
  info[ 7] = (unsigned char)( w>>24);

  info[ 8] = (unsigned char)( h    );
  info[ 9] = (unsigned char)( h>> 8);
  info[10] = (unsigned char)( h>>16);
  info[11] = (unsigned char)( h>>24);

  info[20] = (unsigned char)( sizeData    );
  info[21] = (unsigned char)( sizeData>> 8);
  info[22] = (unsigned char)( sizeData>>16);
  info[23] = (unsigned char)( sizeData>>24);  

  ofstream stream;
  stream.open (fn);
  stream.write( (char*)file, sizeof(file) );
  stream.write( (char*)info, sizeof(info) );

  unsigned char pad[3] = {0,0,0};
  
  for(size_t i = 0; i < h; i++) {
    size_t k = h-1-i;
    for(size_t j = 0; j < w; j++ ) {
      unsigned char pixel[3];
      if (nChannels == 3) {
        if (inf) {
          assert( 0 <= a[rcc_to_idx(k, j, 0)].inf && a[rcc_to_idx(k, j, 0)].inf <= 1 );
          assert( 0 <= a[rcc_to_idx(k, j, 1)].inf && a[rcc_to_idx(k, j, 1)].inf <= 1 );
          assert( 0 <= a[rcc_to_idx(k, j, 2)].inf && a[rcc_to_idx(k, j, 2)].inf <= 1 );         
          pixel[2] = (unsigned char) round(a[rcc_to_idx(k, j, 0)].inf * 255);
          pixel[1] = (unsigned char) round(a[rcc_to_idx(k, j, 1)].inf * 255);
          pixel[0] = (unsigned char) round(a[rcc_to_idx(k, j, 2)].inf * 255);
        } else {
          assert( 0 <= a[rcc_to_idx(k, j, 0)].sup && a[rcc_to_idx(k, j, 0)].sup <= 1 );
          assert( 0 <= a[rcc_to_idx(k, j, 1)].sup && a[rcc_to_idx(k, j, 1)].sup <= 1 );
          assert( 0 <= a[rcc_to_idx(k, j, 2)].sup && a[rcc_to_idx(k, j, 2)].sup <= 1 );
          pixel[2] = (unsigned char) round(a[rcc_to_idx(k, j, 0)].sup * 255);
          pixel[1] = (unsigned char) round(a[rcc_to_idx(k, j, 1)].sup * 255);
          pixel[0] = (unsigned char) round(a[rcc_to_idx(k, j, 2)].sup * 255);
        }
      } else {
        size_t m = rcc_to_idx(k, j, 0);
        if (inf) {
          if (!(0 <= a[rcc_to_idx(k, j, 0)].inf && a[rcc_to_idx(k, j, 0)].inf <= 1)) {
            cout << k << " " << j << " " << a[rcc_to_idx(k, j, 0)].inf << endl;
          }
          assert( 0 <= a[rcc_to_idx(k, j, 0)].inf && a[rcc_to_idx(k, j, 0)].inf <= 1 );
          pixel[2] = (unsigned char) round(a[m].inf * 255);
          pixel[1] = (unsigned char) round(a[m].inf * 255);
          pixel[0] = (unsigned char) round(a[m].inf * 255);
        } else {
          if (!(0 <= a[rcc_to_idx(k, j, 0)].sup && a[rcc_to_idx(k, j, 0)].sup <= 1)) {
            cout << k << " " << j << " " << a[rcc_to_idx(k, j, 0)].sup << endl;
          }
          assert( 0 <= a[rcc_to_idx(k, j, 0)].sup && a[rcc_to_idx(k, j, 0)].sup <= 1 );
          pixel[2] = (unsigned char) round(a[m].sup * 255);
          pixel[1] = (unsigned char) round(a[m].sup * 255);
          pixel[0] = (unsigned char) round(a[m].sup * 255);
        }
      }
      stream.write((char*)pixel, 3);
    }
    stream.write((char*)pad, padSize);
  }
}


void ImageImpl::zero() {
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, k);
        a[m] = {0,0};
      }
    }
  }
}

void ImageImpl::roundToInt() {
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, k);
        if (!a[m].is_empty()) { 
          cout << m << endl;
          a[m] = roundIntervalToInt(a[m]);
        }
      }
    }
  }
}

void ImageImpl::clip(const float min_val, const float max_val) {
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, k);
        if (!a[m].is_empty()) { 
          a[m] = {min(max(min_val, a[m].inf), max_val), min(max(min_val, a[m].sup), max_val)};
        }
      }
    }
  }
}



ImageImpl* ImageImpl::threshold(const float val) {
  ImageImpl* ret = new ImageImpl(nRows, nCols, nChannels);
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, k);
        if (!a[m].is_empty() && a[m].inf >= val) {
          ret->a[m] = {1, 1};
        } else {
          ret->a[m] = {0, 0};
        }
      }
    }
  }
  return ret;
}




ImageImpl* ImageImpl::center_crop(const size_t size) const {
  assert(size % 2 == 0);
  assert(this->nCols % 2 == 0);
  assert(this->nRows % 2 == 0);
  assert(this->nCols >= size);
  assert(this->nRows >= size);
  const size_t diffCols = this->nCols - size;
  const size_t diffRows = this->nRows - size;
  ImageImpl *ret = new ImageImpl(size, size, this->nChannels);
  for (size_t nChannel = 0; nChannel < this->nChannels; ++nChannel) {
    for (size_t nRow = 0; nRow < size; ++nRow) {
      for (size_t nCol = 0; nCol < size; ++nCol) {
        const size_t r = nRow + diffRows / 2;
        const size_t c = nCol + diffCols / 2;
        const size_t m = ret->rcc_to_idx(nRow, nCol, nChannel);
        const size_t n = this->rcc_to_idx(r, c, nChannel);
        //cout << m << " " << n << endl;
        ret->a[m] = this->a[n];
      }
    }
  }  
  // cout << " not here" << endl;
  return ret;
}

ImageImpl* ImageImpl::resize(const size_t new_nRows, const size_t new_nCols, const bool roundToInt) const {
  ImageImpl *ret = new ImageImpl(new_nRows, new_nCols, this->nChannels);
  for (size_t nChannel = 0; nChannel < ret->nChannels; ++nChannel) {
    for (size_t nRow = 0; nRow < ret->nRows; ++nRow) {
      for (size_t nCol = 0; nCol < ret->nCols; ++nCol) {
        Coordinate<float> new_coord = getCoordinate(nRow * this->nRows / (float) ret->nRows,
                                                    nCol * this->nCols / (float) ret->nCols, 
                                                    nChannel);
        size_t m = ret->rcc_to_idx(nRow, nCol, nChannel);
        Interval out = BilinearInterpolation(new_coord, *this);
        if (roundToInt) {
          out = roundIntervalToInt(out);
        }
        ret->a[m] = out;
        //cout << m << " "<< nRow << " " << nCol << " " << nChannel<< " " << ret->a[m] << endl;
      }
    }
  }
  return ret;
}

ImageImpl* ImageImpl::resize(const size_t new_nRows, const size_t new_nCols, const bool roundToInt, const size_t size) const {
  assert(size % 2 == 0);
  //assert(this->nCols % 2 == 0);
  //assert(this->nRows % 2 == 0);
  assert(new_nRows >= size);
  assert(new_nCols >= size);
  const size_t diffCols = new_nCols - size;
  const size_t diffRows = new_nRows - size;
  ImageImpl *ret = new ImageImpl(size, size, this->nChannels);
  for (size_t nChannel = 0; nChannel < this->nChannels; ++nChannel) {
    for (size_t nRow = 0; nRow < size; ++nRow) {
      for (size_t nCol = 0; nCol < size; ++nCol) {
        const size_t r = nRow + diffRows / 2;
        const size_t c = nCol + diffCols / 2;
        const size_t m = ret->rcc_to_idx(nRow, nCol, nChannel);
        Coordinate<float> new_coord = getCoordinate(r * this->nRows / (float) new_nRows,
                                                    c * this->nCols / (float) new_nCols, 
                                                    nChannel);
        Interval out = BilinearInterpolation(new_coord, *this);
        if (roundToInt) {
          out = roundIntervalToInt(out);
        }
        ret->a[m] = out;        
      }
    }
  }  
  
  return ret;
}


vector<ImageImpl*> ImageImpl::split_color_channels() const {
  vector<ImageImpl*> channels(nChannels);
  if (nChannels == 1) {
    channels[0] = new ImageImpl(*this);
  } else {
    for (size_t nChannel = 0; nChannel < nChannels; ++nChannel) {
      ImageImpl *im = new ImageImpl(nRows, nCols, 1);
      for (size_t i = 0; i < nRows; ++i) {
        for (size_t j = 0; j < nCols; ++j) {
          size_t m = im->rcc_to_idx(i, j, 0);
          size_t n = rcc_to_idx(i, j, nChannel);
          im->a[m] = a[n];
        }
      }
      channels[nChannel] = im;
    }
  }
  return channels; 
}


ImageImpl* ImageImpl::customVingette(const ImageImpl* vingette) const {
  ImageImpl *ret = new ImageImpl(this->nRows, this->nCols, this->nChannels);
  assert(vingette->nRows == nRows);
  assert(vingette->nCols == nCols);
  assert(vingette->nChannels == nChannels);
  for (size_t nChannel = 0; nChannel < this->nChannels; ++nChannel) {
    for (size_t nRow = 0; nRow < this->nRows; ++nRow) {
      for (size_t nCol = 0; nCol < this->nCols; ++nCol) {
        const size_t m = rcc_to_idx(nRow, nCol, nChannel);
        if (vingette->a[m].sup == 0)
          ret->a[m] = a[m];
        else
          ret->a[m] = {0, 0};
      }
    }
  }
  return ret;
};

ImageImpl* ImageImpl::widthVingette(const ImageImpl* vingette, const float treshold) const {
  ImageImpl *ret = new ImageImpl(this->nRows, this->nCols, this->nChannels);
  assert(vingette->nRows == nRows);
  assert(vingette->nCols == nCols);
  assert(vingette->nChannels == nChannels);
  for (size_t nChannel = 0; nChannel < this->nChannels; ++nChannel) {
    for (size_t nRow = 0; nRow < this->nRows; ++nRow) {
      for (size_t nCol = 0; nCol < this->nCols; ++nCol) {
        const size_t m = rcc_to_idx(nRow, nCol, nChannel);
        if ((vingette->a[m].sup - vingette->a[m].inf) <= treshold)
          ret->a[m] = a[m];
        else
          ret->a[m] = {0, 0};
      }
    }
  }
  return ret;
};




ImageImpl* ImageImpl::transform(std::function<Coordinate<Interval>(const Coordinate<float>)> T,
                                const bool roundToInt) const {
  ImageImpl *ret = new ImageImpl(this->nRows, this->nCols, this->nChannels);
  for (size_t nChannel = 0; nChannel < this->nChannels; ++nChannel) {
    for (size_t nRow = 0; nRow < this->nRows; ++nRow) {
      for (size_t nCol = 0; nCol < this->nCols; ++nCol) {
        Interval out = BilinearInterpolation(T(this->getCoordinate(nRow, nCol, nChannel)), *this);
        if (roundToInt) {
          out = roundIntervalToInt(out);
        }
        ret->a[ret->rcc_to_idx(nRow, nCol, nChannel)] = out; 
      }
    }
  }
  return ret;
};

ImageImpl* ImageImpl::rotate(const Interval& params, const bool roundToInt) const {
  auto T = [&](const Coordinate<float> it) -> Coordinate<Interval>{
    return Rotation(it, params);
  };
  return transform(T, roundToInt);
};


ImageImpl* ImageImpl::translate(const Interval& dx, const Interval& dy, const bool roundToInt) const {
  auto T = [&](const Coordinate<float> it) -> Coordinate<Interval>{
    return Translation(it, dx, dy);
  };
  return transform(T, roundToInt);
};


vector<float> ImageImpl::bilinear_coefficients(int x_iter, int y_iter, float x, float y) const {
  assert(x_iter <= x && x <= x_iter + 2 && y_iter <= y && y <= y_iter + 2);
  float a_alpha = (x_iter + 2 - x) * (y_iter + 2 - y) / 4;
  float a_beta = (x_iter + 2 - x) * (y - y_iter) / 4;
  float a_gamma = (x - x_iter) * (y_iter + 2 - y) / 4;
  float a_delta = (x - x_iter) * (y - y_iter) / 4;
  return {a_alpha, a_beta, a_gamma, a_delta};
}


bool ImageImpl::inverseRot(ImageImpl** out_ret,
                           ImageImpl** out_vingette,
                           const Interval gamma,
                           const size_t refinements,
                           const bool addIntegerError,
                           const bool toIntegerValues,
                           const bool computeVingette) const {
  auto T = [&](Coordinate<Interval> coord){
    Coordinate<Interval> coord_out = Rotation(coord, gamma);
    return coord_out;
  };
  auto Tinv = [&](Coordinate<Interval> coord){
    Coordinate<Interval> coord_out = Rotation(coord, -gamma);
    return coord_out;
  };

  assert(out_ret != 0);
  if(computeVingette) assert(out_vingette != 0);
  bool isEmpty = inverse(out_ret,
                         out_vingette,
                         T,
                         Tinv,
                         addIntegerError,
                         toIntegerValues,
                         refinements,
                         computeVingette);

  assert(out_ret != 0);
  assert(*out_ret != 0);
  if(computeVingette) {
    assert(out_vingette != 0);
    assert(*out_vingette != 0);
  }
  return isEmpty;
}

bool ImageImpl::inverseTranslation(ImageImpl** out_ret,
                                   ImageImpl** out_vingette,
                                   const Interval dx,
                                   const Interval dy,
                                   const size_t refinements,
                                   const bool addIntegerError,
                                   const bool toIntegerValues,
                                   const bool computeVingette) const {
  auto T = [&](Coordinate<Interval> coord){
    Coordinate<Interval> coord_out = Translation(coord, dx, dy);
    return coord_out;
  };
  auto Tinv = [&](Coordinate<Interval> coord){
    Coordinate<Interval> coord_out = Translation(coord, -dx, -dy);
    return coord_out;
  };
  
  assert(out_ret != 0);
  if(computeVingette) assert(out_vingette != 0);
  bool isEmpty = inverse(out_ret,
                         out_vingette,
                         T,
                         Tinv,
                         addIntegerError,
                         toIntegerValues,
                         refinements,
                         computeVingette);
  assert(out_ret != 0);
  assert(*out_ret != 0);
  if(computeVingette) {
    assert(out_vingette != 0);
    assert(*out_vingette != 0);
  }
  return isEmpty;
}


ostream& operator<<(ostream& os, const ImageImpl& img)
{
  for (size_t k = 0; k < img.nChannels; ++k) {
    os << "Channel " << k << ":" << endl;
    for (size_t i = 0; i < img.nRows; ++i) {
      for (size_t j = 0; j < img.nCols; ++j) {
        size_t m = img.rcc_to_idx(i, j, k);
        os << img.a[m] << " ";
      }
      os << endl;
    }
    os << endl;
  }
  return os;
}



bool ImageImpl::inverse(ImageImpl **out,
                        ImageImpl **vingette,
                        const std::function<Coordinate<Interval>(Coordinate<Interval>)> T,
                        const std::function<Coordinate<Interval>(Coordinate<Interval>)> invT,
                        const bool addIntegerError,
                        const bool toIntegerValues,
                        const size_t nrRefinements,
                        const bool computeVingette) const {
  
  
  ImageImpl *ret = new ImageImpl(this->nRows, this->nCols, this->nChannels);
  ImageImpl *constraints = 0;
  bool isEmpty = false;
  assert(out != 0);
  if (computeVingette) {
    assert(vingette != 0);
    *vingette = new ImageImpl(this->nRows, this->nCols, this->nChannels);    
  }
  
  for(size_t it = 0; it < (1 + nrRefinements); ++it) {
    bool refine = it > 0;
    bool lastIt = (it == nrRefinements);

    if (refine) {
      ImageImpl *tmp = ret;
      ret = constraints;
      constraints = tmp;
      if (ret == 0) ret = new ImageImpl(this->nRows, this->nCols, this->nChannels);
    }  
    
    for (size_t nRow = 0; nRow < this->nRows; ++nRow) 
      for (size_t nCol = 0; nCol < this->nCols; ++nCol) {
        bool e = invert_pixel_on_bounding_box(nRow,
                                              nCol,
                                              ret,
                                              refine,
                                              constraints,
                                              T,
                                              invT,
                                              addIntegerError,
                                              lastIt && toIntegerValues,
                                              lastIt && computeVingette,
                                              (lastIt && computeVingette) ? *vingette : 0);
        isEmpty = isEmpty || e;        
        if (isEmpty) {nRow = this->nRows; nCol=this->nCols;} // break loops
      }
    if (isEmpty) break;
  }

  if (constraints != 0) delete constraints;
  *out = ret;
  return isEmpty;
}


bool ImageImpl::invert_pixel_on_bounding_box(const size_t nRow,
                                             const size_t nCol,
                                             ImageImpl *ret,
                                             const bool refine,
                                             const ImageImpl *constraints,
                                             const std::function<Coordinate<Interval>(Coordinate<Interval>)> T,
                                             const std::function<Coordinate<Interval>(Coordinate<Interval>)> invT,
                                             const bool addIntegerError,
                                             const bool toIntegerValues,
                                             const bool computeVingette,
                                             ImageImpl *vingette) const {
  assert(ret != 0);
  if (refine) assert(constraints != 0);
  Coordinate<float> coord = getCoordinate(nRow, nCol, 0);
  Coordinate<Interval> coordI({coord.x-2, coord.x+2}, {coord.y-2, coord.y+2}, 0);
  Coordinate<Interval> coordI_pre = invT(coordI);

  // pixels to consider
  size_t parity_x = (nCols - 1) % 2;
  size_t parity_y = (nRows - 1) % 2;
  int lo_x, hi_x, lo_y, hi_y;
  tie(lo_x, hi_x, lo_y, hi_y) = calculateBoundingBox(coordI_pre.x, coordI_pre.y, parity_x, parity_y);


  //size_t m = rcc_to_idx(nRow, nCol, 0);
  //printf("%ld %ld %f %f %f %f %f %f %d %d %d %d\n", nRow, nCol, coordI_pre.x.inf, coordI_pre.x.sup, coordI_pre.y.inf, coordI_pre.y.sup, a[m].inf, a[m].sup, lo_x, hi_x, lo_y, hi_y);

  Interval inv_p[nChannels];
  for (size_t chan = 0; chan < this->nChannels; ++chan)
    inv_p[chan] = Interval(0, 1);

  for (int x_iter = lo_x; x_iter <= hi_x; x_iter += 2) 
    for (int y_iter = lo_y; y_iter <= hi_y; y_iter += 2) {
      Coordinate<Interval> coord_iter({(float)x_iter, (float)x_iter}, {(float)y_iter, (float)y_iter}, 0.0f );
      Coordinate<Interval> coord_iterT = T(coord_iter);           

      if (coord_iterT.x.meet(coordI.x).is_empty() || coord_iterT.y.meet(coordI.y).is_empty()) continue;

      Interval p[nChannels];
      int r_iter, c_iter;
      tie(r_iter, c_iter) = getRc(x_iter, y_iter);
      if (0 <= r_iter && r_iter < signed(nRows) &&
          0 <= c_iter && c_iter < signed(nCols)) {
        for (size_t chan = 0; chan < this->nChannels; ++chan) {
          p[chan] = valueAt(x_iter, y_iter, chan, Interval(0, 1));

          if (addIntegerError) {
            //this assumes that the pixel_values * 255 are integers in [0, 255]
            //and that the rounding mode is arithmetic rounding (and not for example floorfing)
            assert(p[chan].inf == p[chan].sup);
            float deltaUp = (0.5f - Constants::EPS)/255.0f;
            float deltaDown = 0.5f / 255.0f;
            p[chan] = Interval(p[chan].inf - deltaDown, p[chan].sup + deltaUp).meet({0, 1});
          }
        }
              
      } else {
        for (size_t chan = 0; chan < this->nChannels; ++chan)
          p[chan] = Interval(0, 1);
      }
      Interval inv_corners[nChannels];
      for (Corner c : {Corner::upper_left, Corner::lower_left, Corner::upper_right, Corner::lower_right}) {
        Interval inv_corner[nChannels];
        invert_pixel(c,
                     coord,
                     coord_iterT,
                     inv_corner,
                     p,
                     constraints,
                     refine,
                     false); // debug
        for (size_t chan = 0; chan < this->nChannels; ++chan) {
          inv_corners[chan] = inv_corners[chan].join(inv_corner[chan]);
        }
      }

      for (size_t chan = 0; chan < this->nChannels; ++chan) {
         
        // printf("%d %d %d %d %d [%f %f] [%f %f] [%f %f]\n", (int)nRow, (int)nCol, x_iter, y_iter, (int)chan, inv_p[chan].inf, inv_p[chan].sup,
        //        inv_corners[chan].inf, inv_corners[chan].sup,
        //        (inv_p[chan].meet(inv_corners[chan])).inf,
        //        (inv_p[chan].meet(inv_corners[chan])).sup);
        
        inv_p[chan] = inv_p[chan].meet(inv_corners[chan]);
      }
    }
  bool isEmpty = false;
  for (size_t chan = 0; chan < this->nChannels; ++chan) {
    size_t m = rcc_to_idx(nRow, nCol, chan);
    if (toIntegerValues && !inv_p[chan].is_empty()) {
      float lower = inv_p[chan].inf;
      float upper = inv_p[chan].sup;

      if (lower > 0) {
        for(size_t k = 0; k < 255; k ++) {
          if(pixelVals[k] < lower && lower <= pixelVals[k+1] ) {
            lower = pixelVals[k+1];
            break;
          }
        }
      }

      assert(lower >= inv_p[chan].inf);          
      if(upper < 1) {
        for(size_t k = 256; k >= 1; k--) {
          if(pixelVals[k] > upper && upper >= pixelVals[k]) {
            upper = pixelVals[k];
            break;
          }
        }
      }
      assert(upper <= inv_p[chan].sup);

            
      if (lower > upper) {
        inv_p[chan] = Interval();
      } else {
        inv_p[chan] = Interval(lower, upper);
      }
    }
    ret->a[m] = inv_p[chan];
    if (computeVingette) {
      if ((ret->a[m].sup - ret->a[m].inf) > 0.3f)
        vingette->a[m] = Interval(1, 1);
      else
        vingette->a[m] = Interval(0, 0);
    }
    isEmpty = isEmpty || ret->a[m].is_empty(); 
  }  
  return isEmpty;
}



void ImageImpl::invert_pixel(const Corner center,
                             const Coordinate<float>& coord,
                             const Coordinate<Interval>& coord_iterT,
                             Interval* out,
                             const Interval* pixel_values,
                             const ImageImpl* constraints,
                             const bool refine,
                             const bool debug) const {

  Interval x_box, y_box;
  float x_ll, y_ll;
  vector<tuple<float, float>> corners;
  
  switch(center) {
  case lower_right:
    x_ll = coord.x;
    y_ll = coord.y;
    x_box = coord_iterT.x.meet(Interval(coord.x, coord.x+2));
    y_box = coord_iterT.y.meet(Interval(coord.y, coord.y+2));
    if (!refine) corners.push_back({x_box.sup, y_box.sup});
    break;
  case upper_right:
    x_ll = coord.x;
    y_ll = coord.y - 2;
    x_box = coord_iterT.x.meet(Interval(coord.x, coord.x+2));
    y_box = coord_iterT.y.meet(Interval(coord.y-2, coord.y));
    if (!refine) corners.push_back({x_box.sup, y_box.inf});
    break;
  case lower_left:
    x_ll = coord.x - 2;
    y_ll = coord.y;
    x_box = coord_iterT.x.meet(Interval(coord.x-2, coord.x));
    y_box = coord_iterT.y.meet(Interval(coord.y, coord.y+2));
    if (!refine) corners.push_back({x_box.inf, y_box.sup});
    break;
  case upper_left:
    x_ll = coord.x - 2;
    y_ll = coord.y - 2;
    x_box = coord_iterT.x.meet(Interval(coord.x-2, coord.x));
    y_box = coord_iterT.y.meet(Interval(coord.y-2, coord.y));
    if (!refine) corners.push_back({x_box.inf, y_box.inf});
    break;
  default:
    assert(false);
  };

  if (debug) printf("%f %f %f %f %f %f %f %f %f %f\n", x_ll, y_ll, x_box.inf, x_box.sup, y_box.inf, y_box.sup, coord_iterT.x.inf, coord_iterT.x.sup, coord_iterT.y.inf, coord_iterT.y.sup);
  
  assert(out != 0);
  for (size_t chan = 0; chan < this->nChannels; ++chan)
    out[chan] = Interval();
   
  if (x_box.is_empty() || y_box.is_empty()) {
    return;
  }

 assert(pixel_values != 0);
    
 if (refine) {
   assert(constraints != 0);
   corners.push_back({x_box.inf, y_box.inf});
   corners.push_back({x_box.sup, y_box.inf});
   corners.push_back({x_box.inf, y_box.sup});
   corners.push_back({x_box.sup, y_box.sup});
 }
  
 for (auto c : corners) {
   float x_coord = get<0>(c), y_coord = get<1>(c);
   Interval ret[nChannels];
   //printf("%f %f %f\n", x_ll, x_coord, x_ll + 2);
   vector<float> coef =  bilinear_coefficients(x_ll, y_ll, x_coord, y_coord);
   if (debug) printf("coefs %f %f %f %f\n", coef[0], coef[1], coef[2], coef[3]);
   if (coef[center] == 0) {
     for (size_t chan = 0; chan < this->nChannels; ++chan)
       ret[chan] = Interval(0, 1);
   } else {
     for (size_t chan = 0; chan < this->nChannels; ++chan) {
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
         for (size_t chan = 0; chan < this->nChannels; ++chan) {
           Interval con = constraints->valueAt(x_, y_, chan, {0, 1});
           if (debug) printf("con [%f %f] %d \n", con.inf, con.sup, con.is_empty());
           ret[chan] = ret[chan] - coef[k] * con;
           if (debug) printf("%d [%f %f]\n", (int)k, ret[chan].inf, ret[chan].sup);
         }
       }
     } else {
       float f = 0;
       for(size_t k = 0; k < 4; ++k) {
         if (k == center) continue;
         f += coef[k];
       }
       Interval tmp = f * Interval(0,1);
       for (size_t chan = 0; chan < this->nChannels; ++chan) {
         ret[chan] = ret[chan] - tmp;
         if (debug) printf("[%f %f]\n", ret[chan].inf, ret[chan].sup);
       }
     }

     for (size_t chan = 0; chan < this->nChannels; ++chan) {
       ret[chan] = ret[chan] / coef[center];
       if (debug) printf("%f [%f %f]\n", coef[center], ret[chan].inf, ret[chan].sup);
     }
   }
   for (size_t chan = 0; chan < this->nChannels; ++chan) {
     out[chan] = out[chan].join(ret[chan]);
   }
 }
}



ImageImpl* ImageImpl::erode() {
  ImageImpl* ret = new ImageImpl(nRows, nCols, nChannels);
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        const size_t m = rcc_to_idx(i, j, k);
        Interval val = a[m];
        for(int di=-1; di <= 1; ++di)
          for(int dj=-1; dj <= 1; ++dj) {
            int ii = (signed)i + di;
            int jj = (signed)j + dj;
            if (0 <= ii && ii < nRows && 0 <= jj && jj < nCols) {
              const size_t mm = rcc_to_idx((unsigned)ii, (unsigned)jj, k);
              Interval other = a[mm];
              if (other.inf <= val.inf && other.sup <= val.sup)
                val = other;
            }
          }
        ret->a[m] = val;
      }
    }
  }
}


ImageImpl* ImageImpl::fill (const float value) const {
  ImageImpl* ret = new ImageImpl(nRows, nCols, nChannels);
  for (size_t k = 0; k < nChannels; ++k) {
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        size_t m = rcc_to_idx(i, j, k);
        ret->a[m] = {value, value};
      }
    }
  }
  return ret;
}
