#include "pytorch_cuda_helper.hpp"
#include "bilinear_interpolate.cuh"

void deform_conv_shape_check(Tensor input, Tensor offset, Tensor* gradOutput,
                             Tensor weight, int kH, int kW, int dH, int dW,
                             int padH, int padW, int dilationH, int dilationW,
                             int group, int deformable_group) {
  TORCH_CHECK(
      weight.ndimension() == 4,
      "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, but got: %s",
      weight.ndimension());

  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  TORCH_CHECK(kW > 0 && kH > 0,
             "kernel size should be greater than zero, but got kH: %d kW: %d",
              kH, kW);

  TORCH_CHECK((weight.size(2) == kH && weight.size(3) == kW),
              "kernel size should be consistent with weight, ",
              "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d",
               kH, kW, weight.size(2), weight.size(3));

  TORCH_CHECK(dW > 0 && dH > 0,
              "stride should be greater than zero, but got dH: %d dW: %d", dH,
              dW);
  TORCH_CHECK(
      dilationW > 0 && dilationH > 0,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH, dilationW);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  TORCH_CHECK(ndim == 3 || ndim == 4,
              "3D or 4D input tensor expected but got: %s", ndim);

  long nInputPlane = weight.size(1) * group;
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);
  long nOutputPlane = weight.size(0);
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  TORCH_CHECK(nInputPlane % deformable_group == 0,
      "input channels must divide deformable group size");
  if (outputWidth < 1 || outputHeight < 1)
    AT_ERROR(
         "Given input size: (%ld x %ld x %ld). "
         "Calculated output size: (%ld x %ld x %ld). Output size is too small",
         nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
         outputWidth);

  TORCH_CHECK(input.size(1) == nInputPlane,
    "invalid number of input planes, expected: %d, but got: %d",
     nInputPlane, input.size(1));

  TORCH_CHECK((inputHeight >= kH && inputWidth >= kW),
       "input image is smaller than kernel");

  TORCH_CHECK(
      (offset.size(2) == outputHeight && offset.size(3) == outputWidth),
      "invalid spatial size of offset, expected height: %d width: %d, but "
      "got height: %d width: %d",
      outputHeight, outputWidth, offset.size(2), offset.size(3));

  TORCH_CHECK((offset.size(1) == deformable_group * 2 * kH * kW),
      "invalid number of channels of offset");

  if (gradOutput != NULL) {
    TORCH_CHECK(
        gradOutput->size(dimf) == nOutputPlane,
        "invalid number of gradOutput planes, expected: %d, but got: %d",
         nOutputPlane, gradOutput->size(dimf));

     TORCH_CHECK(
          (gradOutput->size(dimh) == outputHeight &&
           gradOutput->size(dimw) == outputWidth),
          "invalid size of gradOutput, expected height: %d width: %d , but "
          "got height: %d width: %d",
          outputHeight, outputWidth, gradOutput->size(dimh),
          gradOutput->size(dimw));
  }
}

template <typename T>
__device__ T deformable_im2col_bilinear(const T *input, const int data_width,
                                        const int height, const int width, T h,
                                        T w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = input[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = input[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = input[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = input[h_high * data_width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
__global__ void deformable_im2col_gpu_kernel(
    const int n, const T *data_im, const T *data_offset, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int output_height,
    const int output_width, T *data_col) {

  CUDA_1D_KERNEL_LOOP(index, n) {
    // data_col shape: (cin * kw * kh, batch * oh * ow)
    const int w_out = index % output_width;
    const int h_out = (index / output_width) % output_height;
    const int b = (index / output_width / output_height) % batch_size;
    const int c = index / output_width / output_height / batch_size;
    const int c_k = c * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c / channel_per_deformable_group;

    const int h_in = h_out * stride_h - pad_h;
    const int w_in = w_out * stride_w - pad_w;

    T* data_col_ptr = data_col
                      + c_k * (batch_size * output_height * output_width)
                      + b * output_height * output_width
                      + h_out * output_width + w_out;

    const T* data_im_ptr = data_im + (b * num_channels + c) * height * width;

    // data_offset shape: (b, deformable_group * 2 * kw *kh, oh, ow)
    const T* data_offset_ptr = data_offset
                               + ((((b * deformable_group
                                     + deformable_group_index) * 2 * kernel_h * kernel_w
                                     + 0) * output_height
                                     + h_out) * output_width
                                     + w_out);

    for (int i=0; i < kernel_h; ++i) {
      for (int j=0; j < kernel_w; ++j) {
        const T offset_h = data_offset_ptr[
              (2 * (i * kernel_w + j) + 0) * output_height * output_width];
        const T offset_w = data_offset_ptr[
              (2 * (i * kernel_w + j) + 1) * output_height * output_width];
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        T val = T(0);
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          //val = deformable_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
          val = bilinear_interpolate(data_im_ptr, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * output_height * output_width;
      }  // for kernel_w
    }  // for kernel_h
  }  // CUDA_1D_KERNEL_LOOP
}  // deformable_im2col_gpu_kernel

void deformable_im2col(Tensor data_im, Tensor data_offset,
                       const int in_channels,
                       const int input_height, const int input_width,
                       const int output_height, const int output_width,
                       const int ksize_h, const int ksize_w,
                       const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deformable_group,
                       Tensor data_col) {

  int num_kernels = in_channels * output_height * output_width * parallel_imgs;
  int channel_per_deformable_group = in_channels / deformable_group;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "deformable_im2col_gpu_kernel", [&] {
        deformable_im2col_gpu_kernel<scalar_t>
        <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, data_im.data_ptr<scalar_t>(), data_offset.data_ptr<scalar_t>(),
            input_height, input_width, ksize_h, ksize_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, parallel_imgs, in_channels,
            deformable_group, output_height, output_width,
            data_col.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}


template <typename T>
__global__ void deformable_col2im_gpu_kernel(
    const int n, const T *data_col, const T *data_offset, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    T *grad_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    // columns shape: (cin * kh * kw, batch_size * oh * ow)
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const T *data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const T offset_w = data_offset_ptr[data_offset_w_ptr];
    const T cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const T cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const T cur_top_grad = data_col[index];
    T* offset_grad_im = grad_im + (b * channels + c) * height * width;
    BIPreCalc<T> pc;
    pre_calc_bilinear_interpolate(height, width, cur_inv_h_data, cur_inv_w_data, &pc);

    if (pc.p1 >=0 && pc.p2 >= 0 && pc.p3 >=0 && pc.p4 >= 0) {
          atomicAdd(offset_grad_im + pc.p1, cur_top_grad * pc.w1);
          atomicAdd(offset_grad_im + pc.p2, cur_top_grad * pc.w2);
          atomicAdd(offset_grad_im + pc.p3, cur_top_grad * pc.w3);
          atomicAdd(offset_grad_im + pc.p4, cur_top_grad * pc.w4);
    }
  }
}

template <typename T>
__device__ T get_coordinate_weight(T argmax_h, T argmax_w, const int height,
                                   const int width, const T *im_data,
                                   const int data_width, const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}


template <typename T>
__global__ void deformable_col2im_coord_gpu_kernel(
    const int n, const T *data_col, const T *data_im, const T *data_offset,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int offset_channels, const int deformable_group, const int height_col,
    const int width_col, T *grad_offset) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    // offset_channels = 2 * ksize_h * ksize_w * deformable_group
    // num_kernels = height_col * width_col * 2 * ksize_h * ksize_w * deformable_group * parallel_imgs;
    // channel_per_deformable_group = channels / deformable_group;

    // loop based on offset
    // offset shape: (b, deformable_group * kh * kw * 2, oh, ow)
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;

    const int bp_dir = c % 2;
    const int j = (c / 2) % kernel_w;
    const int i = (c / 2 / kernel_w) % kernel_h;
    const int deformable_group_index = c / (2 * kernel_h * kernel_w);

    // columns shape: (cin * kh * kw, batch_size * oh * ow)
    const T *data_col_ptr = data_col
                            + deformable_group_index
                              * channel_per_deformable_group
                              * kernel_h * kernel_w
                              * batch_size * width_col * height_col
                            + b * height_col * width_col + h * width_col + w;
    // input shape: (b, c, h, w)
    const T *data_im_ptr = data_im
                           + (b * deformable_group + deformable_group_index)
                           * channel_per_deformable_group * height * width;
    // offset shape: (b, deformable_group * kh * kw * 2, oh, ow)
    const T *data_offset_ptr = data_offset
                               + (b * deformable_group + deformable_group_index)
                               * 2 * kernel_h * kernel_w * height_col * width_col
                               + h * width_col + w;

    T val = 0;
    for (int cnt = 0; cnt < channel_per_deformable_group; cnt += 1) {
      const int col_pos = (cnt * kernel_h * kernel_w + (i * kernel_w + j))
                          * batch_size * height_col * width_col;

      int w_out = w;
      int h_out = h;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          ((i * kernel_w + j) * 2) * height_col * width_col;
      const int data_offset_w_ptr =
          ((i * kernel_w + j) * 2 + 1) * height_col * width_col;
      const T offset_h = data_offset_ptr[data_offset_h_ptr];
      const T offset_w = data_offset_ptr[data_offset_w_ptr];
      T inv_h = h_in + i * dilation_h + offset_h;
      T inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
        inv_h = inv_w = -2;
      const T weight = get_coordinate_weight(inv_h, inv_w, height, width,
                                             data_im_ptr + cnt * height * width,
                                             width, bp_dir);
      val += weight * data_col_ptr[col_pos];
    }
    grad_offset[index] = val;
  }
}

void deformable_col2im(Tensor data_col, Tensor data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deformable_group,
                       Tensor grad_im) {
  // todo: make sure parallel_imgs is passed in correctly
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

        deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                       THREADS_PER_BLOCK, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_offset_, channels, height, width,
            ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
            dilation_w, channel_per_deformable_group, parallel_imgs,
            deformable_group, height_col, width_col, grad_im_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void deformable_col2im_coord(
    Tensor data_col, Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, Tensor grad_offset) {
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w *
                    deformable_group * parallel_imgs;
  // int channel_per_deformable_group =
  //     channels * ksize_h * ksize_w / deformable_group;

  int channel_per_deformable_group =
      channels / deformable_group;


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_coord_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data_ptr<scalar_t>();

        deformable_col2im_coord_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_im_, data_offset_, channels, height,
            width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group, parallel_imgs,
            2 * ksize_h * ksize_w * deformable_group, deformable_group,
            height_col, width_col, grad_offset_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}



void DeformConvForwardCUDAKernelLauncher(Tensor input, Tensor weight,
                                         Tensor offset, Tensor output,
                                         Tensor columns, Tensor ones, int kW,
                                         int kH, int dW, int dH, int padW,
                                         int padH, int dilationW, int dilationH,
                                         int group, int deformable_group,
                                         int im2col_step) {

  deform_conv_shape_check(input, offset, NULL, weight, kH, kW, dH, dW, padH,
                          padW, dilationH, dilationW, group, deformable_group);

  at::DeviceGuard guard(input.device());

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input.unsqueeze_(0);
    offset.unsqueeze_(0);
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  output = output.view({batchSize / im2col_step, im2col_step,
                        nOutputPlane, outputHeight, outputWidth});

  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  input = input.view({batchSize / im2col_step, im2col_step,
                      nInputPlane, inputHeight, inputWidth});

  offset = offset.view({batchSize / im2col_step, im2col_step,
                       deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  Tensor output_buffer = at::zeros({batchSize / im2col_step, group,
                                    nOutputPlane / group, im2col_step,
                                    outputHeight, outputWidth},
                                    output.options());

  for (int elt = 0; elt < batchSize / im2col_step; ++elt) {
    deformable_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                      inputWidth, outputHeight, outputWidth, kH, kW, padH,
                      padW, dH, dW, dilationH, dilationW, im2col_step,
                      deformable_group, columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group,
                          weight.size(1), weight.size(2), weight.size(3)});

    for (int g = 0; g < group; ++g) {
      // output_buffer[elt][g] = output_buffer[elt][g]
      //                             .flatten(1)
      //                             .addmm_(weight[g].flatten(1), columns[g])
      //                             .view_as(output_buffer[elt][g]);
      output_buffer[elt][g] = weight[g].flatten(1).mm(columns[g])
                                  .view_as(output_buffer[elt][g]);

    }
    columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  output_buffer = output_buffer.view({output_buffer.size(0), output_buffer.size(1) * output_buffer.size(2),
                                      output_buffer.size(3), output_buffer.size(4), output_buffer.size(5)});
  output_buffer.transpose_(1, 2);
  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
  }
}  // DeformConvForwardCUDAKernelLauncher

// void DeformConvBackwardInputCUDAKernelLauncher(
//     Tensor input, Tensor offset, Tensor gradOutput, Tensor gradInput,
//     Tensor gradOffset, Tensor weight, Tensor columns, int kW, int kH,
//     int dW, int dH, int padW, int padH, int dilationW, int dilationH,
//     int group, int deformable_group, int im2col_step) {
//   // weight: (cout, cin / g * kh * kw) -> (g, cout / g, cin / g * kh * kw)
//   // columns: (cin * kh * kw, batch * oh * ow) -> (g, cin / g * kh * kw, batch * oh * ow)

//   at::DeviceGuard guard(input.device());

//   int batch = 1;
//   if (input.ndimension() == 3) {
//     // Force batch
//     batch = 0;
//     input = input.view({1, input.size(0), input.size(1), input.size(2)});
//     offset = offset.view({1, offset.size(0), offset.size(2), offset.size(3)});
//     gradOutput = gradOutput.view(
//         {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
//   }

//   long batchSize = input.size(0);
//   long nInputPlane = input.size(1);
//   long inputHeight = input.size(2);
//   long inputWidth = input.size(3);

//   long nOutputPlane = weight.size(0);
//   long outputWidth =
//       (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
//   long outputHeight =
//       (inputHeight + 2 * padW - (dilationH * (kH - 1) + 1)) / dH + 1;

//   gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
//   // why columns need to be buffered? tbd...
//   columns = at::zeros(
//       {nInputPlane * kW * kH, im2col_step * outputHeight * outputHeight},
//       input.options());

//   // tbd: fix here, im2col_step
//   gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,
//                                 nOutputPlane, outputHeight, outputWidth});
//   gradOutput.transpose_(1, 2);

//   for (int elt = 0; elt < batchSize / im2col_step, elt++) {
//     // divide into groups

//     columns = columns.view({group, columns.size(0) / group, columns.size(1)});
//     weight = weight.view({group, weight.size(0) / group, weight.size(1), weight.size(2), weight.size(3)});
//     gradOutput = gradOutput.view({gradOutput.size(0), group, gradOutput.size(1) / g, gradOutput.size});


//   }


void DeformConvBackwardInputCUDAKernelLauncher(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradInput,
    Tensor gradOffset, Tensor weight, int kW, int kH, int dW,
    int dH, int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, int im2col_step) {
  deform_conv_shape_check(input, offset, &gradOutput, weight, kH, kW, dH, dW,
                          padH, padW, dilationH, dilationW, group,
                          deformable_group);
  at::DeviceGuard guard(input.device());

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2)});
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth = gradOutput.size(2);
  long outputHeight = gradOutput.size(3);

  TORCH_CHECK((offset.size(0) == batchSize), 3, "invalid batch size of offset");
  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  Tensor grad_columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  // change order of grad output
  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,
                                nOutputPlane, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);
  gradOutput.view({
      gradOutput.size(0), group, gradOutput.size(1) / group,
      gradOutput.size(2) * gradOutput.size(3) * gradOutput.size(4)});

  weight = weight.view({group, weight.size(0) / group,
                        weight.size(1) * weight.size(2) * weight.size(3)});


  gradInput = gradInput.view({batchSize / im2col_step, im2col_step, nInputPlane,
                              inputHeight, inputWidth});
  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  gradOffset = gradOffset.view({batchSize / im2col_step, im2col_step,
                                deformable_group * 2 * kH * kW, outputHeight,
                                outputWidth});
  offset =
      offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    // divide into groups
    grad_columns = grad_columns.view(
        {group, grad_columns.size(0) / group, grad_columns.size(1)});

    for (int g = 0; g < group; g++) {
      grad_columns[g] = grad_columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                                     gradOutput[elt][g].flatten(1), 0.0f, 1.0f);
    }

    grad_columns =
        grad_columns.view({grad_columns.size(0) * grad_columns.size(1), grad_columns.size(2)});

    deformable_col2im_coord(grad_columns, input[elt], offset[elt], nInputPlane,
                            inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW, im2col_step, deformable_group,
                            gradOffset[elt]);

    deformable_col2im(grad_columns, offset[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                      dilationW, im2col_step, deformable_group, gradInput[elt]);
  }

  gradOutput.view({batchSize / im2col_step, nOutputPlane,
      im2col_step, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});
  weight = weight.view({nOutputPlane, nInputPlane / group, kH, kW});


  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  gradOffset = gradOffset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
    gradOffset =
        gradOffset.view({offset.size(1), offset.size(2), offset.size(3)});
  }
}

void DeformConvBackwardParametersCUDAKernelLauncher(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradWeight,
    int kW, int kH, int dW, int dH, int padW,
    int padH, int dilationW, int dilationH, int group, int deformable_group,
    float scale, int im2col_step) {

  deform_conv_shape_check(input, offset, &gradOutput, gradWeight, kH, kW, dH,
                          dW, padH, padW, dilationH, dilationW, group,
                          deformable_group);
  at::DeviceGuard guard(input.device());

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view(
        {1, input.size(0), input.size(1), input.size(2)});
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradWeight.size(0);
  long outputWidth = gradOutput.size(2);
  long outputHeight = gradOutput.size(3);

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");


  Tensor gradOutputBuffer = at::zeros_like(gradOutput);
  gradOutputBuffer.copy_(gradOutput);
  gradOutputBuffer = gradOutputBuffer.view({batchSize / im2col_step, im2col_step,
                                nOutputPlane, outputHeight, outputWidth});
  gradOutputBuffer.transpose_(1, 2);
  gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.size(0), group, gradOutputBuffer.size(1) / group,
         gradOutputBuffer.size(2) * gradOutputBuffer.size(3) * gradOutputBuffer.size(4)});

  gradWeight = gradWeight.view(
      {group, gradWeight.size(0) / group,
      gradWeight.size(1) * gradWeight.size(2) * gradWeight.size(3)});

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  offset =
      offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  Tensor columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());


  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    deformable_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                      inputWidth, outputHeight, outputWidth, kH, kW, padH,
                      padW, dH, dW, dilationH, dilationW, im2col_step,
                      deformable_group, columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});

    for (int g = 0; g < group; g++) {
      gradWeight[g] = gradWeight[g]
                          .addmm_(gradOutputBuffer[elt][g],
                                  columns[g].transpose(1, 0), 1.0, scale);
    }
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});
  gradWeight = gradWeight.view({nOutputPlane, nInputPlane / group, kH, kW});
  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }
}
