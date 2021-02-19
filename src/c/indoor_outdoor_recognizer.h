/*
 * Copyright (c) Xiaomi. 2021. All rights reserved.
 * Description: This the inoutdoor outdoor recognize algorithm header file.
 *
 */
#ifndef SRC_C_INDOOR_OUTDOOR_RECOGNIZER_H_
#define SRC_C_INDOOR_OUTDOOR_RECOGNIZER_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Create an IndoorOutdoor Recognizer
 * @note
 * @retval Recognizer instance
 */
void *mi_indoor_outdoor_recognizer_new();

/**
 * @brief  Free the recognizer instance
 * @note
 * @param  recognizer: Recognizer instance
 * @retval None
 */
void mi_indoor_outdoor_recognizer_free(void *recognizer);

/**
 * @brief  Initialize the recognizer
 * @note   Can be use to reset a recognizer
 * @param  recognizer: Recognizer instance
 * @retval None
 */
void mi_indoor_outdoor_recognizer_init(void *recognizer);

/**
 * @brief  Process parsed NMEA snr data array
 * @note
 * @param  recognizer: Recongnizer instance
 * @param  timestamp_ms: Current timestamp
 * @param  ids: Visiable satellites id array
 * @param  ids_len: Visiable satellites id array length
 * @param  snrs: Visiable satellites snr array
 * @param  snrs_len: Visiable satellites snr array length
 * @retval True when Recognizer status updated, False otherwise
 */
bool mi_indoor_outdoor_recognizer_process(void *   recognizer,
                                          uint64_t timestamp_ms, uint8_t *ids,
                                          uint8_t ids_len, uint8_t *snrs,
                                          uint8_t snrs_len);

/**
 * @brief  Process parsed NMEA GPGSV string
 * @note
 * @param  *recognizer: Recongnizer instance
 * @param  *nmea_sentence: NMEA GPGSV string
 * @retval True when Recognizer status updated, False otherwise
 */
bool mi_indoor_outdoor_recognizer_process_nmea_str(void *recognizer,
                                                   char *nmea_sentence);

uint8_t mi_indoor_outdoor_recognizer_get_status(void *recognizer);

#ifdef __cplusplus
}
#endif

#endif  // SRC_C_INDOOR_OUTDOOR_RECOGNIZER_H_
