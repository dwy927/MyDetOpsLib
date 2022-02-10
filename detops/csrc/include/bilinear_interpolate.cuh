#ifndef BILINEAR_INTERPOLATE_CUH
#define BILINEAR_INTERPOLATE_CUH

#include "cuda.h"

template <typename T>
struct BIPreCalc {
  int p1, p2, p3, p4;
  T w1, w2, w3, w4;
  T lx, ly, hx, hy;
};

template <typename T>
__device__ void pre_calc_bilinear_interpolate(
    const int height, const int width, T y, T x, BIPreCalc<T>* pc) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    pc->w1 = pc->w2 = pc->w3 = pc->w4 = 0.;
    pc->p1 = pc->p2 = pc->p3 = pc->p4 = -1;
    pc->lx = pc->ly = pc->hx = pc->hy = 0;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y1 = (int)y;
  int x1 = (int)x;
  int y2 = y1 + 1;
  int x2 = x1 + 1;

  if (y1 >= height - 1) {
    y1 = y2 = height - 1;
    y = (T)y1;
  }

  if (x1 >= width - 1) {
    x1 = x2 = width - 1;
    x = (T)x1;
  }

  T ly = y - y1;
  T lx = x - x1;
  T hy = 1. - ly, hx = 1. - lx;

  pc->p1 = y1 * width + x1;
  pc->p2 = y1 * width + x2;
  pc->p3 = y2 * width + x1;
  pc->p4 = y2 * width + x2;
  pc->w1 = hy * hx;
  pc->w2 = hy * lx;
  pc->w3 = ly * hx;
  pc->w4 = ly * lx;
  pc->lx = lx;
  pc->ly = ly;
  pc->hx = hx;
  pc->hy = hy;

  return;
}

template <typename T>
__device__ T bilinear_interpolate(const T* input, const int height,
                                  const int width, T y, T x) {
  BIPreCalc<T> pc;
  pre_calc_bilinear_interpolate(height, width, y, x, &pc);
  return pc.w1 * input[pc.p1] + pc.w2 * input[pc.p2] + pc.w3 * input[pc.p3] + pc.w4 * input[pc.p4];
}

#endif  // BILINEAR_INTERPOLATE_CUH
