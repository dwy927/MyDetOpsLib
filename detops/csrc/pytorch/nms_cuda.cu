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

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  at::cuda::CUDAGuard device_guard(boxes.device());
  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }
  auto order_t = std::get<1>(scores.sort(0, true));
  auto boxes_sorted = boxes.index_select(0, order_t);
  int boxes_num = boxes.size(0);
  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  Tensor mask =
    at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));
  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  nms_cuda<<<blocks, threads, 0, stream>>>(
      boxes_num, iou_threshold, offset, boxes_sorted.data_ptr<float>(),
      (unsigned long long*)mask.data_ptr<int64_t>());

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host =
      (unsigned long long*)mask_cpu.data_ptr<int64_t>();

  std::vector<unsigned long long> remv(col_blocks, 0);

  at::Tensor keep_t = at::zeros(
      {boxes_num}, boxes.options().dtype(at::kBool).device(at::kCPU));
  bool* keep = keep_t.data_ptr<bool>();

  for (int i=0; i < boxes_num; ++i) {
    const int nblock = i / threadsPerBlock;
    const int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep[i] = true;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; ++j) {
        remv[j] |= p[j];
      }
    }
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.masked_select(keep_t.to(at::kCUDA));
}
