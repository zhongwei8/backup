/*
 * Copyright (c) Xiaomi. 2021. All rights reserved.
 * Description: This is the har model header file.
 *
 */
#ifndef MIOT_ALGO_HAR_VERSION_H_
#define MIOT_ALGO_HAR_VERSION_H_

const char *miot_algo_har_get_version_info();
const char *miot_algo_har_get_version_name();

int miot_algo_har_get_version_major();
int miot_algo_har_get_version_minor();
int miot_algo_har_get_version_patch();

#endif  // MIOT_ALGO_HAR_VERSION_H_
