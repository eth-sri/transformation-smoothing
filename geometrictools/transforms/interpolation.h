// #include "domains/polyhedra.h"
#include "utils/utilities.h"
#include "utils/image_cpu.h"

#pragma once

Interval BilinearInterpolation(Coordinate<float_t> coord, const ImageImpl& img);

Interval BilinearInterpolation(Coordinate<Interval> coord, const ImageImpl& img);

std::tuple<int, int, int, int> calculateBoundingBox(Interval x, Interval y, size_t parity_x, size_t parity_y);


Interval BiqubicInterpolation(Coordinate<float_t> coord, const ImageImpl& img);

Interval BiqubicInterpolation(Coordinate<Interval> coord, const ImageImpl& img);
