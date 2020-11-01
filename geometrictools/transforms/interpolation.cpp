#include "utils/constants.h"
#include "transforms/interpolation.h"
#include <cmath>
#include <iostream>
#include <tuple>

// #pragma once

using namespace std;

Interval BilinearInterpolation(Coordinate<float_t> coord, const ImageImpl& img){
  return BilinearInterpolation({{coord.x, coord.x}, {coord.y, coord.y}, coord.channel}, img);
}

// Interval InterpolationTransformation::transform(Coordinate<float_t> coord, const ImageImpl<float_t> &img) const {
//     return transform({{coord.x, coord.x}, {coord.y, coord.y}, coord.channel}, img);
// }

/*
  Performs bilinear interpolation of Coordinate coord in the original image.
  Arguments:
    coord: Coordinate which is interpolated (coord.x, coord.y) are its coordinates (both are intervals)
	img: original image
 */
Interval BilinearInterpolation(Coordinate<Interval> coord, const ImageImpl& img) {
  Interval ret;
  size_t parity_x = (img.nCols - 1) % 2;
  size_t parity_y = (img.nRows - 1) % 2;
  int lo_x, hi_x, lo_y, hi_y;

	// computes bounding box of possible concrete coordinates
  tie(lo_x, hi_x, lo_y, hi_y) = calculateBoundingBox(coord.x, coord.y, parity_x, parity_y);
  
  // traverse all interpolation regions
  for (int x1 = lo_x; x1 <= hi_x; x1 += 2) {
    for (int y1 = lo_y; y1 <= hi_y; y1 += 2) {
      // intersect coordinates with interpolation region
      Interval x_box = coord.x.meet(Interval(x1, x1 + 2));
      Interval y_box = coord.y.meet(Interval(y1, y1 + 2));

      // skip if there is no intersection with this region
      if (x_box.is_empty() || y_box.is_empty()) {
        continue;
      }

      // compute coordinates of 4 corners of interpolation region
      float_t alpha, beta, gamma, delta;
      
      alpha = img.valueAt(x1, y1, coord.channel).inf;
      beta = img.valueAt(x1, y1 + 2, coord.channel).inf;
      gamma = img.valueAt(x1 + 2, y1, coord.channel).inf;
      delta = img.valueAt(x1 + 2, y1 + 2, coord.channel).inf;
      
      // use formula for bilinear interpolation to obtain all possible coordinates
      Interval tmp_low = 0.25f * (alpha * (x1 + 2 - x_box) * (y1 + 2 - y_box)
                                  + beta * (x1 + 2 - x_box) * (y_box - y1)
                                  + gamma * (x_box - x1) * (y1 + 2 - y_box)
                                  + delta * (x_box - x1) * (y_box - y1));

      ret = ret.join(tmp_low);

      alpha = img.valueAt(x1, y1, coord.channel).sup;
      beta = img.valueAt(x1, y1 + 2, coord.channel).sup;
      gamma = img.valueAt(x1 + 2, y1, coord.channel).sup;
      delta = img.valueAt(x1 + 2, y1 + 2, coord.channel).sup;

      Interval tmp_high = 0.25f * (alpha * (x1 + 2 - x_box) * (y1 + 2 - y_box)
                                   + beta * (x1 + 2 - x_box) * (y_box - y1)
                                   + gamma * (x_box - x1) * (y1 + 2 - y_box)
                                   + delta * (x_box - x1) * (y_box - y1));
      ret = ret.join(tmp_high);

    }
  }
  assert(!ret.is_empty());
  return ret;
}


std::tuple<int, int, int, int> calculateBoundingBox(Interval x, Interval y, size_t parity_x, size_t parity_y) {
  int lo_x = (int)floor(x.inf - 2*Constants::EPS);
  int hi_x = (int)ceil(x.sup + 2*Constants::EPS);
  int lo_y = (int)floor(y.inf - 2*Constants::EPS);
  int hi_y = (int)ceil(y.sup + 2*Constants::EPS);

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

  return make_tuple(lo_x, hi_x, lo_y, hi_y);
}



Interval BiqubicInterpolation(Coordinate<float_t> coord, const ImageImpl& img){
  return {};//return BiqubicInterpolation({{coord.x, coord.x}, {coord.y, coord.y}, coord.channel}, img);
}

/*
  Performs biqubic interpolation of Coordinate coord in the original image.
  Arguments:
  coord: Coordinate which is interpolated (coord.x, coord.y) are its coordinates (both are intervals)
	img: original image
*/
Interval BiqubicInterpolation(Coordinate<Interval> coord, const ImageImpl& img){
  Interval ret;
  size_t parity_x = (img.nCols - 1) % 2;
  size_t parity_y = (img.nRows - 1) % 2;
  int lo_x, hi_x, lo_y, hi_y;

  // std::cout << std::endl;
  // std::cout << "+++++++++++++++++++++++++++++++++++++++" << std::endl;
  // std::cout << coord.x << std::endl;
  // std::cout << coord.y << std::endl;
  // std::cout << coord.channel << std::endl;
  // std::cout << "+++++++++++++++++++++++++++++++++++++++" << std::endl;


  // computes bounding box of possible concrete coordinates
  tie(lo_x, hi_x, lo_y, hi_y) = calculateBoundingBox(coord.x, coord.y, parity_x, parity_y);

  // std::cout << "lo_x, hi_x, lo_y, hi_y = " << lo_x << "  " << hi_x << "  " << lo_y << "  " << hi_y << std::endl;

  // traverse all interpolation regions
  for (int x1 = lo_x; x1 <= hi_x; x1 += 2) {
    for (int y1 = lo_y; y1 <= hi_y; y1 += 2) {
      // intersect coordinates with interpolation region
      Interval x_box = coord.x.meet(Interval(x1, x1 + 2));
      Interval y_box = coord.y.meet(Interval(y1, y1 + 2));

      // ############## printing ##############
      // std::cout << "########################" << std::endl;
      // std::cout << "x1 = " << x1 << std::endl;
      // std::cout << "y1 = " << y1 << std::endl;
      // std::cout << "x_box = " << x_box << std::endl;
      // std::cout << "y_box = " << y_box << std::endl;
      // ############## printing ##############

      // skip if there is no intersection with this region
      if (x_box.is_empty() || y_box.is_empty()) {
        // std::cout << "continue" << std::endl;
        continue;
      }
      // ############## printing ##############
      // std::cout << "########################" << std::endl;
      // ############## printing ##############

      // assert(x_box.inf == x_box.sup);
      // assert(y_box.inf == y_box.sup);

      // Find pixel values (intervals) at the corresponding positions
      Interval p[4][4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          // std::cout << "coordinates: x = " << x1 + 2 * (i - 1) << "  y = " <<  y1 + 2 * (j - 1) << std::endl;
          
          // increasing first index: matrix down -> y up
          // increasing second index: matrix right -> x up
          p[j][i] = img.valueAt(x1 + 2 * (i - 1), y1 + 2 * (j - 1), coord.channel);
          // assert(p[j][i].inf == p[j][i].sup);
        }
      }

      // ############## printing ##############
      // std::cout << "p matrix" << std::endl;
      // for (int i = 0; i < 4; ++i) {
      //   std::cout << "{";
      //   for (int j = 0; j < 4; ++j) {
      //     std::cout << p[i][j];
      //     if (j < 3) { std::cout << ", "; }
      //   }
      //   std::cout << "}"; 
      //   if (i < 3) { std::cout << ","; }
      //   std::cout << std::endl;
      // }
      // std::cout << std::endl;
      // ############## printing ##############
      

      // calculate coefficients
      Interval coefficients[4][4] {
        {p[1][1],
            (-p[1][0] + p[1][2])/2.,
            p[1][0] - (5*p[1][1])/2. + 2*p[1][2] - p[1][3]/2.,
            (-p[1][0] + 3*p[1][1] - 3*p[1][2] + p[1][3])/2.},

          {(-p[0][1] + p[2][1])/2.,
              (p[0][0] - p[0][2] - p[2][0] + p[2][2])/4.,
              (-2*p[0][0] + 5*p[0][1] - 4*p[0][2] + p[0][3] + 2*p[2][0] - 5*p[2][1] + 4*p[2][2] - p[2][3])/4.,
              (p[0][0] - 3*p[0][1] + 3*p[0][2] - p[0][3] - p[2][0] + 3*p[2][1] - 3*p[2][2] + p[2][3])/4.},

            {p[0][1] - (5*p[1][1])/2. + 2*p[2][1] - p[3][1]/2.,
                (-2*p[0][0] + 2*p[0][2] + 5*p[1][0] - 5*p[1][2] - 4*p[2][0] + 4*p[2][2] + p[3][0] - p[3][2])/4.,
                (4*p[0][0] - 10*p[0][1] + 8*p[0][2] - 2*p[0][3] - 10*p[1][0] + 25*p[1][1] - 20*p[1][2] + 5*p[1][3] + 8*p[2][0] - 20*p[2][1] + 16*p[2][2] - 4*p[2][3] - 2*p[3][0] + 5*p[3][1] - 4*p[3][2] + p[3][3])/4.,
                (-2*p[0][0] + 6*p[0][1] - 6*p[0][2] + 2*p[0][3] + 5*p[1][0] - 15*p[1][1] + 15*p[1][2] - 5*p[1][3] - 4*p[2][0] + 12*p[2][1] - 12*p[2][2] + 4*p[2][3] + p[3][0] - 3*p[3][1] + 3*p[3][2] - p[3][3])/4.},

              {(-p[0][1] + 3*p[1][1] - 3*p[2][1] + p[3][1])/2.,
                  (p[0][0] - p[0][2] - 3*p[1][0] + 3*p[1][2] + 3*p[2][0] - 3*p[2][2] - p[3][0] + p[3][2])/4.,
                  (-2*p[0][0] + 5*p[0][1] - 4*p[0][2] + p[0][3] + 6*p[1][0] - 15*p[1][1] + 12*p[1][2] - 3*p[1][3] - 6*p[2][0] + 15*p[2][1] - 12*p[2][2] + 3*p[2][3] + 2*p[3][0] - 5*p[3][1] + 4*p[3][2] - p[3][3])/4.,
                  (p[0][0] - 3*p[0][1] + 3*p[0][2] - p[0][3] - 3*p[1][0] + 9*p[1][1] - 9*p[1][2] + 3*p[1][3] + 3*p[2][0] - 9*p[2][1] + 9*p[2][2] - 3*p[2][3] - p[3][0] + 3*p[3][1] - 3*p[3][2] + p[3][3])/4.}
      };

      // ############## printing ##############
      // std::cout << "coefficients:" << std::endl;
      // for (int i = 0; i < 4; ++i) {
      //   std::cout << "{";
      //   for (int j = 0; j < 4; ++j) {
      //     std::cout << coefficients[i][j];
      //     // assert(coefficients[i][j].inf == coefficients[i][j].sup);
      //     if (j < 3) { std::cout << ", "; }
      //   }
      //   std::cout << "}"; 
      //   if (i < 3) { std::cout << ","; }
      //   std::cout << std::endl;
      // }
      // std::cout << std::endl;

      // std::cout << "coefficients * x_box^j * y_box^i:" << std::endl;
      // std::cout << "(x_box - x1)/2 = " << (x_box - x1)/2 << std::endl;
      // std::cout << "(y_box - y1)/2 = " << (y_box - y1)/2 << std::endl;
      // for (int i = 0; i < 4; ++i) {
      //   std::cout << "{";
      //   for (int j = 0; j < 4; ++j) {
      //     std::cout << (coefficients[i][j] * ((x_box - x1)/2).pow(j) * ((y_box - y1)/2).pow(i));
      //     assert((coefficients[i][j] * ((x_box - x1)/2).pow(j) * ((y_box - y1)/2).pow(i)).inf 
      //         == (coefficients[i][j] * ((x_box - x1)/2).pow(j) * ((y_box - y1)/2).pow(i)).sup);
      //     if (j < 3) { std::cout << ", "; }
      //   }
      //   std::cout << "}"; 
      //   if (i < 3) { std::cout << ","; }
      //   std::cout << std::endl;
      // }
      // std::cout << std::endl;
      // ############## printing ##############

      // Bicubic interpolation
      Interval tmp {0,0};
      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) { 
          tmp += coefficients[i][j] * ((x_box - x1)/2).pow(j) * ((y_box - y1)/2).pow(i);
        }
      }
    

      // ############## printing ##############
      // std::cout << std::endl;
      // std::cout << "ret befor join = " << ret << std::endl;
      // std::cout << "tmp = " << tmp << std::endl;
      // ############## printing ##############

      ret = ret.join(tmp);

      // ############## printing ##############
      // std::cout << "ret after join = " << ret << std::endl;
      // std::cout << std::endl;
      // ############## printing ##############
    }
  }

  assert(!ret.is_empty());
  // ############## printing ##############
  // if (ret.inf != ret.sup) {
  //   std::cout << " ret not a point, ret = " << ret << std::endl;
  // }
  // assert(ret.inf == ret.sup);
  // ############## printing ##############

  return ret;

}


// void print_matrix(int[][] mat, int rows, int columns) {
//   for (int i = 0; i < rows; ++i) {
//     std::cout << "{";
//     for (int j = 0; j < columns; ++j) {
//       std::cout << p[i][j].inf;
//       if (j < columns-1) {
//         std::cout << ", ";
//       }
//     }
//     std::cout << "}"; 
//     if (i < rows-1) {
//       std::cout << ",";
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;
// }
