/*
 * Copyright (c) Xiaomi. 2020. All rights reserved.
 * Description: This the har detector algorithm header file.
 *
 */
#ifndef HAR_C_HAR_DETECTOR_HAR_DETECTOR_H_
#define HAR_C_HAR_DETECTOR_HAR_DETECTOR_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

enum {
  MI_HAR_UNKNOWN = 0,
  MI_HAR_WALK,
  MI_HAR_RUN,
  MI_HAR_ROWING,
  MI_HAR_ELLIPTICAL,
  MI_HAR_BIKE,
  MI_HAR_NUM_TYPES,
};

struct mi_har_detector;

/* Creare a new activity detector struct */
/**
 * just create a new activity detector
 **/
struct mi_har_detector *mi_har_detector_new();

/* delete and free the activity detector struct */
/**
 * delete and free the activity detector struct
 **/
void mi_har_detector_free(struct mi_har_detector *det);

/* Initialization of the activity detector algorithm */
/**
 * parameter to choose difference algos
 * to initialize static variabels or others before algo running
 **/
void mi_har_detector_init(struct mi_har_detector *det, const float thd,
                          const uint32_t vote_len);

/* main body to process the har model's output probilities */
/**
 * parameter: har model prediction
 * the processed result maybe stored in static variables
 **/
int8_t mi_har_detector_process(struct mi_har_detector *det, const float *probs,
                               const uint8_t num_classes);

/* Get the activity type count */
/**
 * This function will return the index of activity type detected.
 * param None
 * return index of activity type:
 */
uint8_t mi_har_detector_get_activity(struct mi_har_detector *det);

uint32_t mi_har_detector_version();

#ifdef __cplusplus
}
#endif

#endif  // HAR_C_HAR_DETECTOR_HAR_DETECTOR_H_
