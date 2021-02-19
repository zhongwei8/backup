// Copyright (c) Xiaomi. 2021. All rights reserved.
// Author: Farmer Li
// Date: 2021-02-23

#ifndef SRC_C_GPGSV_GPGSV_UTILS_H_
#define SRC_C_GPGSV_GPGSV_UTILS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Parse one integer from string
 * @note   String looks like: "26,00,000,26,1*55", this function will return 26;
 *  return will be "00,000,26,1*55"
 * @param  str: Data string with seperator is comma
 * @retval Return remains string pointer, NULL when exception occurs
 */
char *gpgsv_read_one_interge(char *str, uint8_t *val);

/**
 * @brief  Parse one satellite's id and snr
 * @note   Such as string: "26,00,000,40,1*55", parse 26 as id, 40 as snr;
 * return will be "1*55"
 * @param  str: Data string with seperator is comma
 * @param  id: Output, id
 * @param  snr: Output, snr
 * @retval Return remains string pointer, NULL when exception occurs
 */
char *gpgsv_get_one_satellite_id_and_snr(char *str, uint8_t *id, uint8_t *snr);

/**
 * @brief  Parse ont NMEA string, return GPGSV info
 * @note   NMEA string looks like:
 *  1607938224367,1607938224363,$GPGSV,1,1,03,16,00,000,26,26,00,000,32,27,00,000,22,1*54
 * @param  str: NMEA string pointer
 * @param  ts: Output, timestamp
 * @param  ids: Output, parsed satellite ids
 * @param  snrs: Output, parsed satellite snrs
 * @param  len: Input ids/snrs length, ids and snrs length should be the same
 * @retval Parsed satellites num, less than or equal to parameter @len
 */
uint8_t gpgsv_parse_nmea_str(char *str, uint64_t *ts, uint8_t *ids,
                             uint8_t *snrs, int len);

#ifdef __cplusplus
}
#endif

#endif  // SRC_C_GPGSV_GPGSV_UTILS_H_
