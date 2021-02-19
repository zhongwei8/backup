// Copyright (c) Xiaomi. 2021. All rights reserved.
// Author: Farmer Li
// Date: 2021-02-19
#include <stdlib.h>

#include "activity_recognizer.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

class MIActivityRecognizerPy {
 public:
  MIActivityRecognizerPy(const float thd, const int vote_len) {
    recognizer_ = mi_activity_recognizer_new();
    mi_activity_recognizer_init(recognizer_, thd, vote_len);
  }

  void initAlgo(const float thd, const int vote_len) {
    mi_activity_recognizer_init(recognizer_, thd, vote_len);
  }

  int handleData(float acc_x, float acc_y, float acc_z) {
    return mi_activity_recognizer_process(recognizer_, acc_x, acc_y, acc_z);
  }

  std::vector<float> getPrediction() {
    float *            res = mi_activity_recognizer_get_predicts(recognizer_);
    std::vector<float> v{res, res + 6};
    return v;
  }

  int getActivity() {
    return mi_activity_recognizer_get_activity(recognizer_);
  }

  int getVersion() {
    return mi_activity_recognizer_get_version();
  }

  ~MIActivityRecognizerPy() {
    mi_activity_recognizer_free(recognizer_);
  }

 private:
  void *recognizer_;
};

PYBIND11_MODULE(mi_har_py, m) {
  // optional module docstring
  m.doc() = "Activity Recognizer model python interface";

  // bindings to Worker class
  pybind11::class_<MIActivityRecognizerPy>(m, "MIActivityRecognizerPy")
      .def(pybind11::init<const float, const int>())
      .def("init_algo", &MIActivityRecognizerPy::initAlgo)
      .def("process_data", &MIActivityRecognizerPy::handleData)
      .def("get_probs", &MIActivityRecognizerPy::getPrediction)
      .def("get_type", &MIActivityRecognizerPy::getActivity)
      .def("get_version", &MIActivityRecognizerPy::getVersion);
}
