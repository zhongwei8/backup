/*
 * Copyright (c) Xiaomi. 2020. All rights reserved.
 * Description: This the har detector algorithm header file.
 *
 */
#ifndef HAR_C_HAR_DETECTOR_HAR_DETECTOR_FUNC_H_
#define HAR_C_HAR_DETECTOR_HAR_DETECTOR_FUNC_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

typedef struct vote_candidate {
  uint8_t type;
  float   score;
} vote_candidate_s;

void vote_majority_score(const uint8_t *in_data, const uint32_t cur_idx,
                         const uint32_t vote_len, vote_candidate_s *candidate);

uint8_t argmax_float(const float *probs, const uint8_t num_classes);

#ifdef __cplusplus
}
#endif

#endif  // HAR_C_HAR_DETECTOR_HAR_DETECTOR_FUNC_H_
