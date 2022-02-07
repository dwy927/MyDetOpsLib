#include "pytorch_cpp_helper.hpp"

#ifdef DETOPS_WITH_CUDA
Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int offset);

Tensor nms_cuda(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return NMSCUDAKernelLauncher(boxes, scores, iou_threshold, offset);
}
#endif

Tensor nms_cpu(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }
  auto x1_t = boxes.select(1, 0).contiguous();
  auto y1_t = boxes.select(1, 1).contiguous();
  auto x2_t = boxes.select(1, 2).contiguous();
  auto y2_t = boxes.select(1, 3).contiguous();

  Tensor areas_t = (x2_t - x1_t + offset) * (y2_t - y1_t + offset);
  auto order_t = std::get<1>(scores.sort(0, /*descending=*/true));
  auto nboxes = boxes.size(0);
  Tensor select_t = at::ones({nboxes}, boxes.options().dtype(at::kBool));

  auto select = select_t.data_ptr<bool>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();

  for (int64_t _i = 0; _i < nboxes; ++_i) {
    if (select[_i] == false) continue;
    auto i = order[_i];
    for (int64_t _j = _i + 1; _j < nboxes; ++_j) {
      if (select[_j] == false) continue;
      auto j = order[_j];
      auto xx1 = std::max(x1[i], x1[j]);
      auto xx2 = std::min(x2[i], x2[j]);
      auto yy1 = std::max(y1[i], y1[j]);
      auto yy2 = std::min(y2[i], y2[j]);
      auto w = std::max(0.f, xx2 - xx1 + offset);
      auto h = std::max(0.f, yy2 - yy1 + offset);
      auto inter = w * h;
      auto ovr = inter / (areas[i] + areas[j] - inter);
      if (ovr > iou_threshold) select[_j] = false;
    }
  }
  return order_t.masked_select(select_t);
}

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  if (boxes.device().is_cuda()) {
#ifdef DETOPS_WITH_CUDA
    CHECK_CUDA_INPUT(boxes);
    CHECK_CUDA_INPUT(scores);
    return nms_cuda(boxes, scores, iou_threshold, offset);
#else
    AT_ERROR("nms is not compiled with GPU support");
#endif
  } else {
    CHECK_CPU_INPUT(boxes);
    CHECK_CPU_INPUT(scores);
    return nms_cpu(boxes, scores, iou_threshold, offset);
  }
}
