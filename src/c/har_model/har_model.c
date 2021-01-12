/*
 * Copyright (c) Xiaomi. 2020. All rights reserved.
 * Description: This the har model algorithm source file.
 *
 */
#include "har_model.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "micro_engine_c_interface.h"

#define MI_HAR_MODEL_VERSION (1001)
#define HAR_ACCEL_SAMPLE_RATE (26)
#define HAR_CLASS_NUM (6)
#define INPUT_DIMS 3
#define BUF_LEN (832)  // 8(time) * 26(fs) * 4(channel)
#define CHANNEL_NUM (4)
#define WIN_LEN (208)  // 8 * 26(fs)
#define SHIFT (52)     // 2 * 26(fs)

typedef struct mi_har_model {
  // -------------------------------------------------------
  // data for AI engine
  void *  micro_interpreter_handle;
  float   buffer[BUF_LEN];
  int32_t input_shape[INPUT_DIMS];

  // output
  uint8_t        activity_num;
  void *         output_buffer;
  const int32_t *output_dims;
  uint32_t       output_dims_size;
  float          predicts[HAR_CLASS_NUM];
  uint32_t       cnt;
  uint32_t       idx;
} mi_har_model;

mi_har_model *mi_har_model_new() {
  mi_har_model *mdl = (mi_har_model *)malloc(sizeof(mi_har_model));
  mdl->micro_interpreter_handle = har_cnn_GetMaceMicroEngineHandle();
  return mdl;
}

void mi_har_model_free(mi_har_model *mdl) {
  if (mdl != NULL) {
    free(mdl);
  }
}

void mi_har_model_init(mi_har_model *mdl) {
  mdl->input_shape[0] = 1;
  mdl->input_shape[1] = WIN_LEN;
  mdl->input_shape[2] = CHANNEL_NUM;
  // input data buffer
  for (int32_t i = 0; i < BUF_LEN; ++i) {
    mdl->buffer[i] = 0.f;
  }

  mdl->activity_num = HAR_CLASS_NUM;

  // output data
  for (int32_t i = 0; i < HAR_CLASS_NUM; ++i) {
    mdl->predicts[i] = 0.f;
  }
  mdl->cnt = 0;
  mdl->idx = 0;
}

int8_t mi_har_model_process(mi_har_model *mdl, float acc_x, float acc_y,
                            float acc_z) {
  mdl->buffer[mdl->idx * 4] = acc_x;
  mdl->buffer[mdl->idx * 4 + 1] = acc_y;
  mdl->buffer[mdl->idx * 4 + 2] = acc_z;
  mdl->buffer[mdl->idx * 4 + 3] =
      sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z);
  mdl->cnt++;
  mdl->idx++;
  int8_t updated = 0;
  if (mdl->cnt >= WIN_LEN && mdl->idx >= WIN_LEN) {
    har_cnn_RegisterInputData(mdl->micro_interpreter_handle, 0,
                              (void *)mdl->buffer, mdl->input_shape);

    har_cnn_Interpret(mdl->micro_interpreter_handle);

    har_cnn_GetInterpretResult(mdl->micro_interpreter_handle, 0,
                               &(mdl->output_buffer), &(mdl->output_dims),
                               &(mdl->output_dims_size));
    float *output = (float *)mdl->output_buffer;
    for (int32_t i = 0; i < mdl->activity_num; i++) {
      mdl->predicts[i] = output[i];
    }
    for (int32_t i = SHIFT; i < WIN_LEN; ++i) {
      for (int32_t j = 0; j < CHANNEL_NUM; ++j) {
        mdl->buffer[CHANNEL_NUM * (i - SHIFT) + j] =
            mdl->buffer[CHANNEL_NUM * i + j];
      }
    }
    mdl->idx = WIN_LEN - SHIFT;
    updated = 1;
  }
  return updated;
}

float *mi_har_model_get_predicts(struct mi_har_model *mdl) {
  return mdl->predicts;
}

uint32_t mi_har_model_version() {
  return MI_HAR_MODEL_VERSION;
}
