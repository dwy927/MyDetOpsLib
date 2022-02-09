#include "pytorch_cpp_helper.hpp"

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset);

void roi_pool_forward(Tensor input, Tensor rois, Tensor output, Tensor argmax,
                      int pooled_height, int pooled_width,
                      float spatial_scale);

void roi_pool_backward(Tensor grad_output, Tensor rois, Tensor argmax,
                       Tensor grad_input, int pooled_height,
                       int pooled_width);

void roi_align_forward(Tensor input, Tensor rois, Tensor output,
                       Tensor argmax_y, Tensor argmax_x, int aligned_height,
                       int aligned_width, float spatial_scale,
                       int sampling_ratio, int pool_mode, bool aligned);

void roi_align_backward(Tensor grad_output, Tensor rois, Tensor argmax_y,
                        Tensor argmax_x, Tensor grad_input, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, int pool_mode, bool aligned);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "nms (CPU/CUDA)", py::arg("boxes"),
      py::arg("scores"), py::arg("iou_threshold"), py::arg("offset"));

  m.def("softnms", &softnms, "softnms (CPU)", py::arg("boxes"),
      py::arg("scores"), py::arg("dets"), py::arg("iou_threshold"),
      py::arg("sigma"), py::arg("min_score"), py::arg("method"),
      py::arg("offset"));

  m.def("roi_pool_forward", &roi_pool_forward, "roi_pool_forward (GPU)",
      py::arg("input"), py::arg("rois"), py::arg("output"),
      py::arg("argmax"), py::arg("pooled_height"), py::arg("pooled_width"),
      py::arg("spatial_scale"));

  m.def("roi_pool_backward", &roi_pool_backward, "roi_pool_backward (GPU)",
      py::arg("grad_output"), py::arg("rois"), py::arg("argmax"),
      py::arg("grad_input"), py::arg("pooled_height"), py::arg("pooled_width"));

  m.def("roi_align_forward", &roi_align_forward, "roi_align_forward (CPU/GPU)",
      py::arg("input"), py::arg("rois"), py::arg("output"),
      py::arg("argmax_y"), py::arg("argmax_x"), py::arg("aligned_height"),
      py::arg("aligned_width"), py::arg("spatial_scale"),
      py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));

  m.def("roi_align_backward", &roi_align_backward,
      "roi_align_backward (CPU/GPU)",
      py::arg("grad_output"), py::arg("rois"), py::arg("argmax_y"),
      py::arg("argmax_x"), py::arg("grad_input"), py::arg("aligned_height"),
      py::arg("aligned_width"), py::arg("spatial_scale"),
      py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
}
