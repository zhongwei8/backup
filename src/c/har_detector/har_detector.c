/*
 * Copyright (c) Xiaomi. 2020. All rights reserved.
 * Description: This the har detector algorithm source file.
 *
 */
#include "har_detector.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include "har_detector_func.h"

#define MI_HAR_DETECTOR_VERSION (1001)
#define BUF_LEN 15

// Struct har detector
struct mi_har_detector {
  uint8_t  buffer[BUF_LEN];
  uint32_t vote_len;

  uint8_t cur_predict;
  float   score_thd;

  uint32_t cnt;
  uint32_t idx;
};

void mi_har_detector_init(struct mi_har_detector *det, const float thd,
                          uint32_t vote_len) {
  det->cnt = 0;
  det->idx = 0;
  if (vote_len <= 0) {
    det->vote_len = 1;
  } else {
    det->vote_len = vote_len;
  }
  for (int32_t i = 0; i < BUF_LEN; ++i) {
    det->buffer[i] = MI_HAR_UNKNOWN;
  }
  det->score_thd = thd;
  det->cur_predict = MI_HAR_UNKNOWN;
}

struct mi_har_detector *mi_har_detector_new() {
  struct mi_har_detector *det =
      (struct mi_har_detector *)malloc(sizeof(struct mi_har_detector));
  return det;
}

void mi_har_detector_free(struct mi_har_detector *det) {
  if (det != NULL) {
    free(det);
  }
}

void vote_majority_score(const uint8_t *in_data, const uint32_t cur_idx,
                         const uint32_t vote_len, vote_candidate_s *candidate) {
  uint32_t counts[MI_HAR_NUM_TYPES] = {0, 0, 0, 0, 0, 0};
  for (int i = 0; i < vote_len; ++i) {
    int32_t idx = cur_idx - i;
    if (cur_idx < i) {
      idx += BUF_LEN;
    }
    uint8_t d = in_data[idx];
    counts[d]++;
  }
  uint32_t max_count = counts[0];
  uint8_t  type = 0;
  for (uint8_t i = 1; i < MI_HAR_NUM_TYPES; ++i) {
    if (counts[i] > max_count) {
      max_count = counts[i];
      type = i;
    }
  }
  candidate->type = type;
  candidate->score = (max_count * 1.0f) / (vote_len * 1.0);
}


uint8_t argmax_float(const float *probs, const uint8_t num_classes) {
  float   p_max = 0;
  uint8_t max_idx = MI_HAR_UNKNOWN;
  for (int i = 0; i < num_classes; ++i) {
    if (p_max < probs[i]) {
      p_max = probs[i];
      max_idx = i;
    }
  }
  return max_idx;
}

int8_t mi_har_detector_process(struct mi_har_detector *det, const float *probs,
                               const uint8_t num_classes) {
  if (num_classes != MI_HAR_NUM_TYPES) {
    return -1;
  }
  uint8_t model_predict = argmax_float(probs, MI_HAR_NUM_TYPES);
  det->buffer[det->idx] = model_predict;
  int8_t update = 0;
  if (det->cnt >= det->vote_len) {
    vote_candidate_s candidate;
    candidate.type = 0;
    candidate.score = 0.f;
    vote_majority_score(det->buffer, det->idx, det->vote_len, &candidate);
    if (candidate.score >= det->score_thd) {
      det->cur_predict = candidate.type;
      update = 1;
    }
  } else {
    if (det->cur_predict != model_predict) {
      det->cur_predict = model_predict;
      update = 1;
    }
  }
  det->cnt++;
  det->idx++;
  if (det->idx >= det->vote_len) {
    det->idx = 0;
  }
  return update;
}

uint8_t mi_har_detector_get_activity(struct mi_har_detector *det) {
  return det->cur_predict;
}

uint32_t mi_har_detector_version() {
  return MI_HAR_DETECTOR_VERSION;
}
