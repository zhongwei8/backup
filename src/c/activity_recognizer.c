// Copyright (c) Xiaomi. 2021. All rights reserved.
// Author: Farmer Li
// Date: 2021-02-19

#include <stddef.h>
#include <stdlib.h>

#include "har_detector/har_detector.h"
#include "har_model/har_model.h"
#include "miot_algo_har_version.h"

typedef struct _MIActivityRecognizer {
  struct mi_har_model *   mdl;
  struct mi_har_detector *det;
  float *                 probs;
  int8_t                  current_type;
} MIActivityRecognizer;

MIActivityRecognizer *mi_activity_recognizer_new() {
  MIActivityRecognizer *recognizer =
      (MIActivityRecognizer *)malloc(sizeof(MIActivityRecognizer));
  recognizer->mdl = mi_har_model_new();
  recognizer->det = mi_har_detector_new();

  return recognizer;
}

void mi_activity_recognizer_free(MIActivityRecognizer *recognizer) {
  if (recognizer != NULL) {
    mi_har_model_free(recognizer->mdl);
    mi_har_detector_free(recognizer->det);
    free(recognizer);
  }
}

void mi_activity_recognizer_init(MIActivityRecognizer *recognizer,
                                 const float thd, const uint32_t vote_len) {
  mi_har_model_init(recognizer->mdl);
  mi_har_detector_init(recognizer->det, thd, vote_len);
}

int8_t mi_activity_recognizer_process(MIActivityRecognizer *recognizer,
                                      float acc_x, float acc_y, float acc_z) {
  int8_t update = mi_har_model_process(recognizer->mdl, acc_x, acc_y, acc_z);
  if (update) {
    recognizer->probs = mi_har_model_get_predicts(recognizer->mdl);
    mi_har_detector_process(recognizer->det, recognizer->probs, HAR_CLASS_NUM);
    recognizer->current_type = mi_har_detector_get_activity(recognizer->det);
  }
  return update;
}

float *mi_activity_recognizer_get_predicts(MIActivityRecognizer *recognizer) {
  return recognizer->probs;
}

int8_t mi_activity_recognizer_get_activity(MIActivityRecognizer *recognizer) {
  return recognizer->current_type;
}

uint32_t mi_activity_recognizer_get_version() {
  uint32_t version = miot_algo_har_get_version_patch();
  version += miot_algo_har_get_version_minor() * 100;
  version += miot_algo_har_get_version_major() * 10000;

  return version;
}
