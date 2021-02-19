// Copyright (c) Xiaomi. 2021. All rights reserved.
// Author: Farmer Li
// Date: 2021-02-23

#include "gpgsv/gpgsv_utils.h"

#include <stddef.h>

#define IS_DIGITAL(x) ((x) >= '0' && (x) <= '9')
#define CHAR_TO_DIGITAL(x) ((x) - '0')

static inline char *gpgsv_skip_k_comma(char *str, uint8_t k) {
  while (k > 0) {
    if (str == NULL || *str == '\0' || *str == '*') {
      return NULL;
    }
    if (*str == ',') {
      k--;
    }
    str++;
  }

  return str;
};

char *gpgsv_read_one_interge(char *str, uint8_t *val) {
  *val = 0;
  char *pc = str;
  while (IS_DIGITAL(*pc)) {
    *val = (*val) * 10 + CHAR_TO_DIGITAL(*pc);
    ++pc;
  }
  if (*pc != '\0') {
    ++pc;
  }

  return pc;
}

char *gpgsv_get_one_satellite_id_and_snr(char *str, uint8_t *id, uint8_t *snr) {
  *id = 0;
  *snr = 255;
  if (str == NULL || *str == '\0') {
    return str;
  }

  // Read the satellite id
  str = gpgsv_read_one_interge(str, id);
  if (str == NULL || *str == '\0') {
    return NULL;
  }

  str = gpgsv_skip_k_comma(str, 2);
  if (str == NULL || *str == '\0') {
    return NULL;
  }

  // Read the satellite snr
  str = gpgsv_read_one_interge(str, snr);

  return str;
}

uint8_t gpgsv_parse_nmea_str(char *str, uint64_t *ts, uint8_t *ids,
                             uint8_t *snrs, int len) {
  char *pc = str;

  // 1) Get timestamp in the front of nmea sentence
  uint64_t timestamp_ms = 0;
  while (IS_DIGITAL(*pc)) {
    timestamp_ms = timestamp_ms * 10 + CHAR_TO_DIGITAL(*pc);
    pc++;
  }
  *ts = timestamp_ms;

  // 2) Find "$GPGSV"
  pc = strstr(pc, "$GPGSV");
  if (pc == NULL) {
    return 0;
  }
  pc = gpgsv_skip_k_comma(pc, 3);
  if (pc == NULL) {
    return 0;
  }
  // 3) Get visiable_satellite_cnt
  uint8_t visiable_satellite_cnt = 0;
  pc = gpgsv_read_one_interge(pc, &visiable_satellite_cnt);
  if (visiable_satellite_cnt <= 0) {
    return 0;
  }

  // 4) Get ids and snrs
  uint8_t id = 0;
  uint8_t snr = 0;
  uint8_t ids_len = 0;
  while (pc != NULL && *pc != '\0') {
    pc = gpgsv_get_one_satellite_id_and_snr(pc, &id, &snr);
    if (id == 0 || snr == 255 || pc == NULL) {
      continue;
    }
    if (ids_len < len) {
      ids[ids_len] = id;
      snrs[ids_len] = snr;
      ids_len++;
    }
  }

  return ids_len;
}
