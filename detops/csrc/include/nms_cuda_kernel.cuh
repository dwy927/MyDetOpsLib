#ifndef NMS_CUDA_KERNEL_CUH
#define NMS_CUDA_KERNEL_CUH
#include "pytorch_cuda_helper.hpp"

const int threadsPerBlock = sizeof(unsigned long long int) * 8;

__device__ inline bool devIoU(const float* const a, const float* const b,
                              const int offset, const float threshold) {

  const float x1 = fmaxf(a[0], b[0]);
  const float y1 = fmaxf(a[1], b[1]);
  const float x2 = fminf(a[2], b[2]);
  const float y2 = fminf(a[3], b[3]);
  const float w = fmax(x2 - x1 + offset, 0.f);
  const float h = fmax(y2 - y1 + offset, 0.f);
  const float inter = w * h;
  const float sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  const float sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  return inter > (sa + sb - inter) * threshold;
}

__global__ void nms_cuda(const int n_boxes, const float iou_threshold,
    const int offset, const float *dev_boxes, unsigned long long* dev_mask) {

  const int row_start = blockIdx.x;
  const int col_start = blockIdx.y;
  const int tid = threadIdx.x;
  if (row_start > col_start) return;
  const int row_size =
    fminf(threadsPerBlock, n_boxes - row_start * threadsPerBlock);
  const int col_size =
    fminf(threadsPerBlock, n_boxes - col_start * threadsPerBlock);


  if (tid >= row_size) return;
  const int cur_box_idx = threadsPerBlock * row_start + tid;
  const float* cur_box = dev_boxes + cur_box_idx * 4;
  unsigned long long int t = 0;
  int start = 0;
  if (row_start == col_start) {start = tid + 1;}
  for (int i = start; i < col_size; ++i) {
    const int ibox_idx = threadsPerBlock * col_start + i;
    const float* ibox = dev_boxes + ibox_idx * 4;
    if (devIoU(cur_box, ibox, offset, iou_threshold)) {
      t |= (1ULL << i);
    }
  }
  dev_mask[cur_box_idx * gridDim.y + col_start] = t;
}

#endif  // NMS_CUDA_KERNEL_CUH
