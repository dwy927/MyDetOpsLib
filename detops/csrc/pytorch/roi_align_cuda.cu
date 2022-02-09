#include "pytorch_cuda_helper.hpp"
#include "bilinear_interpolate.cuh"

template < typename T>
__global__ void roi_align_forward_cuda_kernel(
                const int nthreads, const T* input, const T* rois, T* output,
                T* argmax_y, T* argmax_x, const int pooled_height,
                const int pooled_width, const T spatial_scale,
                const int sampling_ratio,
                const int pool_mode,  // 0 - max pool, 1 - avg pool
                const bool aligned, const int channels, const int height,
                const int width) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
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
    if (!aligned) {  // for backward-compatibility only
      roi_w = std::max(roi_w, (T)1.);
      roi_h = std::max(roi_h, (T)1.);
    }

    const T bin_size_h = static_cast<T>(roi_h) / static_cast<T>(pooled_height);
    const T bin_size_w = static_cast<T>(roi_w) / static_cast<T>(pooled_width);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    const int bin_grid_h = (sampling_ratio > 0)
                            ? sampling_ratio
                            : ceil(roi_h / pooled_height);  // e.g., = 2

    const int bin_grid_w = (sampling_ratio > 0)
                                ? sampling_ratio
                                : ceil(roi_w / pooled_width);  // e.g., = 2

    const T bin_grid_size_h = bin_size_h / static_cast<T>(bin_grid_h);
    const T bin_grid_size_w = bin_size_w / static_cast<T>(bin_grid_w);

    // for avg pool
    const T count = max(bin_grid_h * bin_grid_w, 1);
    T output_val = 0;

    // for max pool
    T maxval = -FLT_MAX;
    T maxidx_y = -1.f, maxidx_x = -1.f;

    for (int iy = 0; iy < bin_grid_h; ++iy) {
      const T y = roi_y1 + ph * bin_size_h + (iy + 0.5) * bin_grid_size_h;
      for (int ix = 0; ix < bin_grid_w; ++ix) {
        const T x = roi_x1 + pw * bin_size_w + (ix + 0.5) * bin_grid_size_w;
        T val = bilinear_interpolate(offset_input, height, width, y, x);
        if (val > maxval) {
          maxval = val;
          maxidx_y = y;
          maxidx_x = x;
        }
        output_val += val;
      }  // for ix
    }  // for iy

    if (pool_mode == 0) {  // max pool
      output[index] = maxval;
      argmax_y[index] = maxidx_y;
      argmax_x[index] = maxidx_x;
    } else if (pool_mode == 1) {  // avg pool
      output[index] = output_val / count;
    }  // if pool_mode
  }  // CUDA_1D_KERNEL_LOOP
}  // roi_align_forward_cuda_kernel

/*** Backward ***/
template <typename T>
__global__ void roi_align_backward_cuda_kernel(
    const int nthreads, const T* grad_output, const T* rois, const T* argmax_y,
    const T* argmax_x, T* grad_input, const int pooled_height,
    const int pooled_width, const T spatial_scale, const int sampling_ratio,
    const int pool_mode,  // 0 - max pool, 1 - avg pool
    const bool aligned, const int channels, const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output.
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    const int roi_batch_ind = offset_rois[0];
    T* offset_grad_input =
        grad_input + (roi_batch_ind * channels + c) * height * width;

    const T grad_output_this_bin = grad_output[index];

    if (pool_mode == 0) {  // max pool
      T y = argmax_y[index], x = argmax_x[index];
      if (y == -1.f) return;
      BIPreCalc<T> pc;
      pre_calc_bilinear_interpolate(height, width, y, x, &pc);
      if (pc.p1 >=0 && pc.p2 >= 0 && pc.p3 >=0 && pc.p4 >= 0) {
        atomicAdd(offset_grad_input + pc.p1, grad_output_this_bin * pc.w1);
        atomicAdd(offset_grad_input + pc.p2, grad_output_this_bin * pc.w2);
        atomicAdd(offset_grad_input + pc.p3, grad_output_this_bin * pc.w3);
        atomicAdd(offset_grad_input + pc.p4, grad_output_this_bin * pc.w4);
      }
    } else if (pool_mode == 1) {  // avg pool
      const T offset = aligned ? (T)0.5: (T)0.0;
      const T roi_x1 = offset_rois[1] * spatial_scale - offset;
      const T roi_y1 = offset_rois[2] * spatial_scale - offset;
      const T roi_x2 = offset_rois[3] * spatial_scale - offset;
      const T roi_y2 = offset_rois[4] * spatial_scale - offset;

      T roi_w = roi_x2 - roi_x1;
      T roi_h = roi_y2 - roi_y1;
      if (!aligned) {  // for backward-compatibility only
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
                                  : ceil(roi_w / pooled_width);  // e.g., = 2

      const T bin_grid_size_h = bin_size_h / static_cast<T>(bin_grid_h);
      const T bin_grid_size_w = bin_size_w / static_cast<T>(bin_grid_w);
      const T count = max(bin_grid_h * bin_grid_w, 1);

      for (int iy = 0; iy < bin_grid_h; ++iy) {
        const T y = roi_y1 + ph * bin_size_h + (iy + 0.5) * bin_grid_size_h;
        for (int ix = 0; ix < bin_grid_w; ++ix) {
          const T x = roi_x1 + pw * bin_size_w + (ix + 0.5) * bin_grid_size_w;
          BIPreCalc<T> pc;
          pre_calc_bilinear_interpolate(height, width, y, x, &pc);
          if (pc.p1 >=0 && pc.p2 >= 0 && pc.p3 >=0 && pc.p4 >= 0) {
            atomicAdd(offset_grad_input + pc.p1, grad_output_this_bin * pc.w1 / count);
            atomicAdd(offset_grad_input + pc.p2, grad_output_this_bin * pc.w2 / count);
            atomicAdd(offset_grad_input + pc.p3, grad_output_this_bin * pc.w3 / count);
            atomicAdd(offset_grad_input + pc.p4, grad_output_this_bin * pc.w4 / count);
          }
        }  // if ix
      }  // if iy
    }  // if pool_mode
  }  // CUDA_1D_KERNEL_LOOP
}  // roi_align_backward_cuda_kernel



void RoIAlignForwardCUDALauncher(Tensor input, Tensor rois, Tensor output,
                                 Tensor argmax_y, Tensor argmax_x,
                                 int aligned_height, int aligned_width,
                                 float spatial_scale, int sampling_ratio,
                                 int pool_mode, bool aligned) {
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ROIAlign_forward_cuda_kernel", [&] {
        roi_align_forward_cuda_kernel<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size, input.data_ptr<scalar_t>(), rois.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(),
            argmax_x.data_ptr<scalar_t>(), aligned_height, aligned_width,
            static_cast<scalar_t>(spatial_scale), sampling_ratio, pool_mode,
            aligned, channels, height, width);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void RoIAlignBackwardCUDALauncher(Tensor grad_output, Tensor rois,
                                  Tensor argmax_y, Tensor argmax_x,
                                  Tensor grad_input, int aligned_height,
                                  int aligned_width, float spatial_scale,
                                  int sampling_ratio, int pool_mode,
                                  bool aligned) {
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "ROIAlign_backward_cuda_kernel", [&] {
        roi_align_backward_cuda_kernel<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size, grad_output.data_ptr<scalar_t>(),
            rois.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(),
            argmax_x.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
            aligned_height, aligned_width, static_cast<scalar_t>(spatial_scale),
            sampling_ratio, pool_mode, aligned, channels, height, width);
      });
}
