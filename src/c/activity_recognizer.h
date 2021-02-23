// Copyright (c) Xiaomi. 2021. All rights reserved.
// Author: Farmer Li
// Date: 2021-02-19

#ifndef SRC_C_ACTIVITY_RECOGNIZER_H_
#define SRC_C_ACTIVITY_RECOGNIZER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct MIActivityRecognizer;

/**
 * @brief  Create an activity recognizer instance.
 * @note
 * @retval MIActivityRecognizer
 */
MIActivityRecognizer *mi_activity_recognizer_new();

/**
 * @brief  Free initialized recognizer instance.
 * @note
 * @param  *recognizer: Activity recognizer instance
 * @retval None
 */
void mi_activity_recognizer_free(MIActivityRecognizer *recognizer);

/**
 * @brief  Initialize an activity recognizer instance.
 * @note   Can be use to reset the instance
 * @param  *recognizer: Activity Recognizer instance
 * @param  thd: Threshold for Activity Recognizer to smooth the results, from 0
 * to 1, 0.8 is recommended
 * @param  vote_len: Results smooth Window length, 15 is recommended
 * @retval None
 */
void mi_activity_recognizer_init(MIActivityRecognizer *recognizer,
                                 const float thd, const uint32_t vote_len);

/**
 * @brief  Process accelerometer data to recognize current activity.
 * @note   Accelerometer sample rate should be 26Hz, recongnize win length is
 * 4s, overlap 2s So when recognizer get 4 seconds data, it will update every 2
 * seconds
 * @param  *recognizer: Activity Recognizer instance
 * @param  acc_x: Accel x axis data
 * @param  acc_y: Accel y axis data
 * @param  acc_z: Accel z axis data
 * @retval 1 when Activity Recognizer updated, 0 otherwise
 */
int8_t mi_activity_recognizer_process(MIActivityRecognizer *recognizer,
                                      float acc_x, float acc_y, float acc_z);

/**
 * @brief  Get **CURRENT** window recognize result
 * @note
 * @param  *recognizer: Activity Recognizer instacne
 * @retval Float array of each activity type's prediction probability
 */
float *mi_activity_recognizer_get_predicts(MIActivityRecognizer *recognizer);

/**
 * @brief  Get smoothed activity recognition result
 * @note
 * @param  *recognizer: Activity Recognizer instance
 * @retval Current recognized activity type
 */
int8_t mi_activity_recognizer_get_activity(MIActivityRecognizer *recognizer);

/**
 * @brief  Get activity recngnition algorithm's version
 * @note   For example, for version v1.2.3, encode as 10000 * 1 + 100 * 2 + 3
 * @retval Encoded version code
 */
uint32_t mi_activity_recognizer_get_version();

#ifdef __cplusplus
}
#endif

#endif  // SRC_C_ACTIVITY_RECOGNIZER_H_
