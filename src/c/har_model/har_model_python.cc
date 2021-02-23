// Copyright 2020 Xiaomi
#include <stdlib.h>

#include "har_model.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

class HarModel {
 public:
  HarModel() {
    mdl_ = mi_har_model_new();
    mi_har_model_init(mdl_);
  }

  void InitAlgo() {
    mi_har_model_init(mdl_);
  }

  int HandleData(float acc_x, float acc_y, float acc_z) {
    return mi_har_model_process(mdl_, acc_x, acc_y, acc_z);
  }

  std::vector<float> GetPrediction() {
    float *            res = mi_har_model_get_predicts(mdl_);
    std::vector<float> v{res, res + 6};
    return v;
  }

  int GetVersion() {
    return mi_har_model_version();
  }

  ~HarModel() {
    mi_har_model_free(mdl_);
  }

 private:
  struct mi_har_model *mdl_;
};

PYBIND11_MODULE(har_model, m) {
  // optional module docstring
  m.doc() = "har model python interface";

  // bindings to Worker class
  pybind11::class_<HarModel>(m, "HarModel")
      .def(pybind11::init<>())
      .def("init_algo", &HarModel::InitAlgo)
      .def("process_data", &HarModel::HandleData)
      .def("get_probs", &HarModel::GetPrediction)
      .def("get_version", &HarModel::GetVersion);
}
