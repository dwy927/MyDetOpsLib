#include "pytorch_cuda_helper.hpp"
#include "bilinear_interpolate.cuh"

/*** forward ***/
template <typename T>
__global__ void deform_roi_pool_forward_cuda_kernel(
    const int nthreads, const T* input, const T* rois, const T* offset,
    T* output, const int pooled_height, const int pooled_width,
    const T spatial_scale, const int sampling_ratio, const T gamma,
    const int channels, const int height, const int width) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output.
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    const int roi_batch_ind = offset_rois[0];

    // -0.5 is for align.
    T roi_x1 = offset_rois[1] * spatial_scale - 0.5;
    T roi_y1 = offset_rois[2] * spatial_scale - 0.5;
    T roi_x2 = offset_rois[3] * spatial_scale - 0.5;
    T roi_y2 = offset_rois[4] * spatial_scale - 0.5;

    T roi_w = roi_x2 - roi_x1;
    T roi_h = roi_y2 - roi_y1;

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

    // Compute roi offset. The only difference with roi_align is here.
    if (offset != NULL) {
      const T* offset_offset = offset + n * 2 * pooled_height * pooled_width + ph * pooled_width + pw;
      T offset_x1 = gamma * roi_w * offset_offset[0];
      T offset_y1 = gamma * roi_h * offset_offset[pooled_height * pooled_width];
      roi_x1 += offset_x1;
      roi_y1 += offset_y1;
    }

    // we do avg pooling inside a bin.
    const T count = max(bin_grid_h * bin_grid_w, 1);
    T output_val = 0;

    for (int iy = 0; iy < bin_grid_h; ++iy) {
      const T y = roi_y1 + ph * bin_size_h + (iy + 0.5) * bin_grid_size_h;
      for (int ix = 0; ix < bin_grid_w; ++ix) {
        const T x = roi_x1 + pw * bin_size_w + (ix + 0.5) * bin_grid_size_w;
        T val = bilinear_interpolate(offset_input, height, width, y, x);
        output_val += val;
      }  // for ix
    }  // for iy
    output[index] = output_val / count;
  }  // CUDA_1D_KERNEL_LOOP
}  // deform_roi_pool_forward_cuda_kernel


void DeformRoIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois,
                                            Tensor offset, Tensor output,
                                            int pooled_height, int pooled_width,
                                            float spatial_scale,
                                            int sampling_ratio, float gamma) {
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "deform_roi_pool_forward_cuda_kernel", [&] {
      deform_roi_pool_forward_cuda_kernel<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size, input.data_ptr<scalar_t>(),
            rois.data_ptr<scalar_t>(), offset.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), pooled_height, pooled_width,
            static_cast<scalar_t>(spatial_scale), sampling_ratio,
            static_cast<scalar_t>(gamma), channels, height, width);
    });
  AT_CUDA_CHECK(cudaGetLastError());
}


/*** backward ***/

template <typename T>
__global__ void deform_roi_pool_backward_cuda_kernel(
    const int nthreads, const T* grad_output, const T* input, const T* rois,
    const T* offset, T* grad_input, T* grad_offset, const int pooled_height,
    const int pooled_width, const T spatial_scale, const int sampling_ratio,
    const T gamma, const int channels, const int height, const int width) {
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
    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    T roi_x1 = offset_rois[1] * spatial_scale - 0.5;
    T roi_y1 = offset_rois[2] * spatial_scale - 0.5;
    T roi_x2 = offset_rois[3] * spatial_scale - 0.5;
    T roi_y2 = offset_rois[4] * spatial_scale - 0.5;

    T roi_w = roi_x2 - roi_x1;
    T roi_h = roi_y2 - roi_y1;

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

    // Compute roi offset. The only difference with roi_align is here.
    if (offset != NULL) {
      const T* offset_offset = offset + n * 2 * pooled_height * pooled_width + ph * pooled_width + pw;
      T offset_x1 = gamma * roi_w * offset_offset[0];
      T offset_y1 = gamma * roi_h * offset_offset[pooled_height * pooled_width];
      roi_x1 += offset_x1;
      roi_y1 += offset_y1;
    }

    const T count = max(bin_grid_h * bin_grid_w, 1);
    const T grad_output_this_bin = grad_output[index] / count;

    for (int iy = 0; iy < bin_grid_h; ++iy) {
      const T y = roi_y1 + ph * bin_size_h + (iy + 0.5) * bin_grid_size_h;
      for (int ix = 0; ix < bin_grid_w; ++ix) {
        const T x = roi_x1 + pw * bin_size_w + (ix + 0.5) * bin_grid_size_w;
        BIPreCalc<T> pc;
        pre_calc_bilinear_interpolate(height, width, y, x, &pc);
        if (pc.p1 >=0 && pc.p2 >= 0 && pc.p3 >=0 && pc.p4 >= 0) {
          atomicAdd(offset_grad_input + pc.p1, grad_output_this_bin * pc.w1);
          atomicAdd(offset_grad_input + pc.p2, grad_output_this_bin * pc.w2);
          atomicAdd(offset_grad_input + pc.p3, grad_output_this_bin * pc.w3);
          atomicAdd(offset_grad_input + pc.p4, grad_output_this_bin * pc.w4);
        }
        if (offset != NULL) {
          T input_1 = offset_input[pc.p1];
          T input_2 = offset_input[pc.p2];
          T input_3 = offset_input[pc.p1];
          T input_4 = offset_input[pc.p2];
          T gx = gamma * roi_w * grad_output_this_bin * (
                  offset_input[pc.p1] * pc.hy * (-1) +
                  offset_input[pc.p2] * pc.hy +
                  offset_input[pc.p3] * pc.ly * (-1) +
                  offset_input[pc.p4] * pc.ly);
          T gy = gamma * roi_w * grad_output_this_bin * (
                  offset_input[pc.p1] * pc.hx * (-1) +
                  offset_input[pc.p2] * pc.lx * (-1) +
                  offset_input[pc.p3] * pc.hx +
                  offset_input[pc.p4] * pc.lx);

          atomicAdd(grad_offset
                    + n * 2 * pooled_height * pooled_width
                    + ph * pooled_width
                    + pw,
                    gx);
          atomicAdd(grad_offset
                    + n * 2 * pooled_height * pooled_width
                    + ph * pooled_width
                    + pw +
                    pooled_height * pooled_width,
                    gy);
        }
      }  // if ix
    }  // if iy
  }  // CUDA_1D_KERNEL_LOOP
}  // deform_roi_pool_backward_cuda_kernel











void DeformRoIPoolBackwardCUDAKernelLauncher(
    Tensor grad_output, Tensor input, Tensor rois, Tensor offset,
    Tensor grad_input, Tensor grad_offset, int pooled_height, int pooled_width,
    float spatial_scale, int sampling_ratio, float gamma) {
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "deform_roi_pool_backward_cuda_kernel", [&] {
      deform_roi_pool_backward_cuda_kernel<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size, grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(), rois.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
            grad_offset.data_ptr<scalar_t>(), pooled_height, pooled_width,
            static_cast<scalar_t>(spatial_scale), sampling_ratio,
            static_cast<scalar_t>(gamma), channels, height, width);
    });
  AT_CUDA_CHECK(cudaGetLastError());
}
