/*
 * Copyright (c) Xiaomi. 2020. All rights reserved.
 * Description: This is the har model header file.
 *
 */
#ifndef HAR_C_HAR_MODEL_HAR_MODEL_H_
#define HAR_C_HAR_MODEL_HAR_MODEL_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

struct mi_har_model;

/* Creare a new activity detector struct */
/**
 * just create a new activity detector
 **/
struct mi_har_model *mi_har_model_new();

/* delete and free the activity detector struct */
/**
 * delete and free the activity detector struct
 **/
void mi_har_model_free(struct mi_har_model *mdl);

/* Initialization of the activity detector algorithm */
/**
 * parameter to choose difference algos
 * to initialize static variabels or others before algo running
 **/
void mi_har_model_init(struct mi_har_model *mdl);

/* main body to process the har model's output probilities */
/**
 * parameter: har model prediction
 * the processed result maybe stored in static variables
 **/
int8_t mi_har_model_process(struct mi_har_model *mdl, float acc_x, float acc_y,
                            float acc_z);

/* Get the har model's predictions */
/**
 * This function will return the probility array activity type detected.
 * param None
 * return index of activity type:
 */
float *mi_har_model_get_predicts(struct mi_har_model *mdl);

uint32_t mi_har_model_version();

#ifdef __cplusplus
}
#endif

#endif  // HAR_C_HAR_MODEL_HAR_MODEL_H_
