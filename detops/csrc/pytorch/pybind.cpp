#include "pytorch_cpp_helper.hpp"

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "nms (CPU/CUDA)", py::arg("boxes"),
      py::arg("scores"), py::arg("iou_threshold"), py::arg("offset"));
}
