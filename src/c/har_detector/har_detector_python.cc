/*
 * Copyright (c) Xiaomi. 2020. All rights reserved.
 * Description: This is the har detector python interface source file.
 *
 */
#include <pybind11/stl.h>
#include <stdlib.h>
#include "har_detector.h"
#include "pybind11/pybind11.h"

class HarDetector {
 public:
  HarDetector(const float thd, const int vote_len) {
    det_ = mi_har_detector_new();
    mi_har_detector_init(det_, thd, vote_len);
  }

  void InitAlgo(const float thd, const int vote_len) {
    mi_har_detector_init(det_, thd, vote_len);
  }

  int HandleProbs(const std::vector<float> &probs, int num_classes) {
    return mi_har_detector_process(det_, probs.data(), num_classes);
  }

  int GetActivityType() {
    return mi_har_detector_get_activity(det_);
  }

  int GetVersion() {
    return mi_har_detector_version();
  }

  ~HarDetector() {
    mi_har_detector_free(det_);
  }

 private:
  struct mi_har_detector *det_;
};

PYBIND11_MODULE(har_detector, m) {
  // optional module docstring
  m.doc() = "har detector python interface";

  // bindings to Worker class
  pybind11::class_<HarDetector>(m, "HarDetector")
      .def(pybind11::init<const float, const int>())
      .def("init_algo", &HarDetector::InitAlgo)
      .def("process_probs", &HarDetector::HandleProbs)
      .def("get_activity_type", &HarDetector::GetActivityType)
      .def("get_version", &HarDetector::GetVersion);
}
