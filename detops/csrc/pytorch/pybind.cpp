#include "pytorch_cpp_helper.hpp"

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "nms (CPU/CUDA)", py::arg("boxes"),
      py::arg("scores"), py::arg("iou_threshold"), py::arg("offset"));

  m.def("softnms", &softnms, "softnms (CPU)", py::arg("boxes"),
      py::arg("scores"), py::arg("dets"), py::arg("iou_threshold"),
      py::arg("sigma"), py::arg("min_score"), py::arg("method"),
      py::arg("offset"));
}
