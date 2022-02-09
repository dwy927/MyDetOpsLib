#include "pytorch_cpp_helper.hpp"

template <typename T>
struct PreCalc {
  PreCalc(int p1, int p2, int p3, int p4, T ww1, T ww2, T ww3, T ww4):
    pos1(p1), pos2(p2), pos3(p3), pos4(p4),
    w1(ww1), w2(ww2), w3(ww3), w4(ww4) {}

  int pos1, pos2, pos3, pos4;
  T w1, w2, w3, w4;
};

template <typename T>
PreCalc<T> bilinear_interpolate(const int height, const int width, T y, T x) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return PreCalc<T>(/*p1*/ -1,
                      /*p2*/ -1,
                      /*p3*/ -1,
                      /*p4*/ -1,
                      /*w1*/ (T)0,
                      /*w2*/ (T)0,
                      /*w3*/ (T)0,
                      /*w4*/ (T)0);
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;
  int x1 = int(x);
  int y1 = int(y);
  int x2 = x1 + 1;
  int y2 = y1 + 1;

  if (y1 >= height - 1) {
    y1 = y2 = height -1;
    y = (T)y1;
  }

  if (x1 >= width - 1) {
    x1 = x2 = width -1;
    x = (T)x1;
  }

  T ly = y - y1;
  T lx = x - x1;
  T hy = 1. - ly;
  T hx = 1. - lx;

  return PreCalc<T>(
              /*p1*/ y1 * width + x1,
              /*p2*/ y1 * width + x2,
              /*p3*/ y2 * width + x1,
              /*p4*/ y2 * width + x2,
              /*w1*/ hy * hx,
              /*w2*/ hy * lx,
              /*w3*/ ly * hx,
              /*w4*/ ly * lx);
}


template <typename T>
std::vector<PreCalc<T>> pre_calc_for_bilinear_interpolate(
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int bin_grid_h, const int bin_grid_w,
    T roi_x1, T roi_y1, T bin_size_h, T bin_size_w, T bin_grid_size_h,
    T bin_grid_size_w) {

  std::vector<PreCalc<T>> pre_calc;
  pre_calc.reserve(pooled_height * pooled_width * bin_grid_h * bin_grid_w);

  for (int ph = 0; ph < pooled_height; ++ph) {
    for (int pw = 0; pw < pooled_width; ++pw) {
      for (int iy = 0; iy < bin_grid_h; ++iy) {
        const T yy =  roi_y1 + ph * bin_size_h +
                      static_cast<T>(iy + .5f) * bin_grid_size_h;
        for (int ix=0; ix < bin_grid_w; ++ix) {
          const T xx =  roi_x1 + pw * bin_size_w +
                        static_cast<T>(ix + .5f) * bin_grid_size_w;

          pre_calc.emplace_back(bilinear_interpolate(height, width, yy, xx));
        }  // for ix
      }  // for iy
    }  // for pw
  }  // for ph

  return pre_calc;
}


template <typename T>
void RoIAlignForward(const int nthreads, const T* input, const T* rois,
                     T* output, T* argmax_y, T* argmax_x,
                     const int pooled_height, const int pooled_width,
                     const T spatial_scale, const int sampling_ratio,
                     const int pool_mode,  // 0 - max pool, 1 - avg pool
                     const bool aligned, const int channels, const int height,
                     const int width) {

  int n_rois = nthreads / pooled_width / pooled_height / channels;
  // (n, c, ph, pw) is an element in the pooled output.
  // came be parallized using omp
  // #pragma omp parallel for num_threads(32)
  for (int n = 0; n < n_rois; ++n) {
    int index_n = n * channels * pooled_height * pooled_width;

    const T* offset_rois = rois + n * 5;
    const int roi_batch_ind = offset_rois[0];

    const T offset = aligned ? (T)0.5: (T)0.0;
    const T roi_x1 = offset_rois[1] * spatial_scale - offset;
    const T roi_y1 = offset_rois[2] * spatial_scale - offset;
    const T roi_x2 = offset_rois[3] * spatial_scale - offset;
    const T roi_y2 = offset_rois[4] * spatial_scale - offset;

    T roi_w = roi_x2 - roi_x1;
    T roi_h = roi_y2 - roi_y1;
    if (aligned) {
      AT_ASSERTM(roi_w >= 0 && roi_h >= 0,
            "ROIs in ROIAlign cannot have non-negative size!");
    } else {  // for backward-compatibility only
      roi_w = std::max(roi_w, (T)1.);
      roi_h = std::max(roi_h, (T)1.);
    }

    const T bin_size_h = static_cast<T>(roi_h) / static_cast<T>(pooled_height);
    const T bin_size_w = static_cast<T>(roi_w) / static_cast<T>(pooled_width);

    const int bin_grid_h = (sampling_ratio > 0)
                            ? sampling_ratio
                            : ceil(roi_h / pooled_height);  // e.g., = 2

    const int bin_grid_w = (sampling_ratio > 0)
                                ? sampling_ratio
                                : ceil(roi_w / pooled_width);

    const T bin_grid_size_h = bin_size_h / static_cast<T>(bin_grid_h);
    const T bin_grid_size_w = bin_size_w / static_cast<T>(bin_grid_w);

    const T count = std::max(bin_grid_h * bin_grid_w, 1);  // e.g., = 4

    // We want to precalculate indices and weights shared by all channels.
    std::vector<PreCalc<T>> pre_calc = pre_calc_for_bilinear_interpolate(
        height, width, pooled_height, pooled_width, bin_grid_h, bin_grid_w,
        roi_x1, roi_y1, bin_size_h, bin_size_w, bin_grid_size_h,
        bin_grid_size_w);

    for (int c = 0; c < channels; ++c) {
      int index_n_c = index_n + c * pooled_height * pooled_width;
      const T* offset_input =
          input + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int index = index_n_c + ph * pooled_height + pw;

          T output_val = 0;
          T maxval = -10000;
          T maxidx_y = -1.f, maxidx_x = -1.f;
          for (int iy = 0; iy < bin_grid_h; ++iy) {
            const T y = roi_y1 + ph * bin_size_h + (iy + 0.5) * bin_grid_size_h;
            for (int ix = 0; ix < bin_grid_w; ++ix) {
              const T x = roi_x1 + pw * bin_size_w + (ix + 0.5) * bin_grid_size_w;
              const PreCalc<T> pc = pre_calc[pre_calc_index];
              pre_calc_index += 1;
              const T val = pc.w1 * offset_input[pc.pos1] +
                            pc.w2 * offset_input[pc.pos2] +
                            pc.w3 * offset_input[pc.pos3] +
                            pc.w4 * offset_input[pc.pos4];
              if (val > maxval) {
                maxval = val;
                maxidx_y = y;
                maxidx_x = x;
              }
              output_val += val;
            }
          }
          if (pool_mode == 0) {  // max pool
            output[index] = maxval;
            argmax_y[index] = maxidx_y;
            argmax_x[index] = maxidx_x;
          } else if (pool_mode == 1) {  // avg pool
            output[index] = output_val / count;
          }  // if
        }  // for pw
      }  // for ph
    }  // for c
  }  // for n
}


template <typename T>
inline void add(T* address, T val) {
  *address += val;
}

template <typename T>
void RoIAlignBackward(const int num_threads, const T* grad_output, const T* rois,
                      const T* argmax_y, const T* argmax_x, T* grad_input,
                      const int pooled_height, const int pooled_width,
                      const T spatial_scale, const int sampling_ratio,
                      const int pool_mode,  // 0 - max pool, 1 - avg pool
                      const bool aligned, const int channels, const int height,
                      const int width, const int n_stride, const int c_stride,
                      const int h_stride, const int w_stride) {
  for (int index=0; index < num_threads; ++index) {
    // (n, c, ph, pw) is an element in the pooled output.
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    const int roi_batch_ind = offset_rois[0];

    const T offset = aligned ? (T)0.5: (T)0.0;
    const T roi_x1 = offset_rois[1] * spatial_scale - offset;
    const T roi_y1 = offset_rois[2] * spatial_scale - offset;
    const T roi_x2 = offset_rois[3] * spatial_scale - offset;
    const T roi_y2 = offset_rois[4] * spatial_scale - offset;

    T roi_w = roi_x2 - roi_x1;
    T roi_h = roi_y2 - roi_y1;
    if (aligned) {
      AT_ASSERTM(roi_w >= 0 && roi_h >= 0,
            "ROIs in ROIAlign cannot have non-negative size!");
    } else {  // for backward-compatibility only
      roi_w = std::max(roi_w, (T)1.);
      roi_h = std::max(roi_h, (T)1.);
    }

    const T bin_size_h = static_cast<T>(roi_h) / static_cast<T>(pooled_height);
    const T bin_size_w = static_cast<T>(roi_w) / static_cast<T>(pooled_width);

    T* offset_grad_input =
      grad_input + ((roi_batch_ind * channels + c) * height * width);

    const T grad_output_this_bin = *(grad_output +
        n * n_stride + c * c_stride + ph * h_stride + pw * w_stride);

    if (pool_mode == 0) {
      // We do max pooling inside a bin;
      T y = argmax_y[index], x = argmax_x[index];
      if (y == -1.f) continue;

      PreCalc<T> pc = bilinear_interpolate(height, width, y, x);
      T g1 = grad_output_this_bin * pc.w1;
      T g2 = grad_output_this_bin * pc.w2;
      T g3 = grad_output_this_bin * pc.w3;
      T g4 = grad_output_this_bin * pc.w4;
      if (pc.pos1 >= 0 && pc.pos2 >= 0 && pc.pos3 >= 0 && pc.pos4 >= 0) {
        add(offset_grad_input + pc.pos1, g1);
        add(offset_grad_input + pc.pos2, g2);
        add(offset_grad_input + pc.pos3, g3);
        add(offset_grad_input + pc.pos4, g4);
      }

    } else if (pool_mode == 1) {
      // We do average pooling inside a bin.
      int bin_grid_h = (sampling_ratio > 0)
                        ? sampling_ratio
                        : ceil(roi_h / pooled_height);  // e.g., = 2
      int bin_grid_w = (sampling_ratio > 0)
                        ? sampling_ratio
                        : ceil(roi_w / pooled_height);  // e.g., = 2

      const T bin_grid_size_h = bin_size_h / static_cast<T>(bin_grid_h);
      const T bin_grid_size_w = bin_size_w / static_cast<T>(bin_grid_w);

      const T count = bin_grid_h * bin_grid_w;  // e.g., = 4
      for (int iy = 0; iy < bin_grid_h; ++iy) {
        const T y = roi_y1 + ph * bin_size_h + (iy + .5f) * bin_grid_size_h;
        for (int ix = 0; ix < bin_grid_w; ++ix) {
          const T x = roi_x1 + pw * bin_size_w + (ix + .5f) * bin_grid_size_w;

          PreCalc<T> pc = bilinear_interpolate(height, width, y, x);

          T g1 = grad_output_this_bin * pc.w1 / count;
          T g2 = grad_output_this_bin * pc.w2 / count;
          T g3 = grad_output_this_bin * pc.w3 / count;
          T g4 = grad_output_this_bin * pc.w4 / count;
          if (pc.pos1 >= 0 && pc.pos2 >= 0 && pc.pos3 >= 0 && pc.pos4 >= 0) {
            add(offset_grad_input + pc.pos1, g1);
            add(offset_grad_input + pc.pos2, g2);
            add(offset_grad_input + pc.pos3, g3);
            add(offset_grad_input + pc.pos4, g4);
          }
        }  // for ix
      }  // for iy
    }  // if pool mode
  }  // for
}  // RoIAlignBackward

void RoIAlignForwardCPULauncher(Tensor input, Tensor rois, Tensor output,
                                      Tensor argmax_y, Tensor argmax_x,
                                      int aligned_height, int aligned_width,
                                      float spatial_scale, int sampling_ratio,
                                      int pool_mode, bool aligned) {
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ROIAlign_forward", [&] {
        RoIAlignForward<scalar_t>(
            output_size, input.data_ptr<scalar_t>(), rois.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(),
            argmax_x.data_ptr<scalar_t>(), aligned_height, aligned_width,
            static_cast<scalar_t>(spatial_scale), sampling_ratio, pool_mode,
            aligned, channels, height, width);
      });
}

void RoIAlignBackwardCPULauncher(Tensor grad_output, Tensor rois,
                                 Tensor argmax_y, Tensor argmax_x,
                                 Tensor grad_input, int aligned_height,
                                 int aligned_width, float spatial_scale,
                                 int sampling_ratio, int pool_mode,
                                 bool aligned) {
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  // get stride values to ensure indexing into gradients is correct.
  int n_stride = grad_output.stride(0);
  int c_stride = grad_output.stride(1);
  int h_stride = grad_output.stride(2);
  int w_stride = grad_output.stride(3);


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "ROIAlign_backward", [&] {
        RoIAlignBackward<scalar_t>(
            output_size, grad_output.data_ptr<scalar_t>(),
            rois.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(),
            argmax_x.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
            aligned_height, aligned_width, static_cast<scalar_t>(spatial_scale),
            sampling_ratio, pool_mode, aligned, channels, height, width,
            n_stride, c_stride, h_stride, w_stride);
      });
}
