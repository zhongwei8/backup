// Copyright (c) Xiaomi. 2021. All rights reserved.
// Author: Zhongwei Tian
// Date: 2021-02-23

#include "indoor_outdoor_recognizer.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "gpgsv/gpgsv_utils.h"

#define THRESHOLD_SATELLITE_CNT (4)
#define THRESHOLD_SNR_SUM (80)
#define RECOGNIZE_WIN_DURARION_MS (4000)

#define SATELLITE_NUM_MAX (32)
#define SINGLE_NEMA_SATELLITE_NUM_MAX (4)

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

typedef enum _INDOOR_OUTDOOR_STATUS {
  UNDIFINE = 0,
  INDOOR = 1,
  OUTDOOR = 2
} IndoorOutdoorStatus;

typedef struct _MIIndoorOutdoorRecognizer {
  uint64_t            update_timestamp_ms;
  uint8_t             parse_ids[SINGLE_NEMA_SATELLITE_NUM_MAX];
  uint8_t             parse_snrs[SINGLE_NEMA_SATELLITE_NUM_MAX];
  uint8_t             snrs[SATELLITE_NUM_MAX];
  uint8_t             sat_cnt;
  uint16_t            sat_snr_sum;
  IndoorOutdoorStatus status;
} MIIndoorOutdoorRecognizer;

void *mi_indoor_outdoor_recognizer_new() {
  MIIndoorOutdoorRecognizer *recognizer =
      (MIIndoorOutdoorRecognizer *)malloc(sizeof(MIIndoorOutdoorRecognizer));
  return (void *)recognizer;
}

void mi_indoor_outdoor_recognizer_free(void *instance) {
  MIIndoorOutdoorRecognizer *recognizer = (MIIndoorOutdoorRecognizer *)instance;

  if (recognizer != NULL) {
    free(recognizer);
  }
}

static inline void reset_recognize_status(
    MIIndoorOutdoorRecognizer *recognizer) {
  for (uint8_t i = 0; i < SATELLITE_NUM_MAX; ++i) {
    recognizer->snrs[i] = 0;
  }
  recognizer->sat_cnt = 0;
  recognizer->sat_snr_sum = 0;
}

void mi_indoor_outdoor_recognizer_init(void *instance) {
  MIIndoorOutdoorRecognizer *recognizer = (MIIndoorOutdoorRecognizer *)instance;
  recognizer->status = UNDIFINE;
  reset_recognize_status(recognizer);
}

static inline void update_gpgsv_info(MIIndoorOutdoorRecognizer *recognizer,
                                     uint8_t id, uint8_t snr) {
  --id;  // Id range from [01, 32] convert to [00, 31]
  recognizer->snrs[id] = snr;
}

static inline void classify(MIIndoorOutdoorRecognizer *recognizer) {
  recognizer->sat_cnt = 0;
  recognizer->sat_snr_sum = 0;
  for (uint8_t i = 0; i < SATELLITE_NUM_MAX; i++) {
    if (recognizer->snrs[i] > 0) {
      recognizer->sat_cnt += 1;
      recognizer->sat_snr_sum += recognizer->snrs[i];
    }
  }

  if (recognizer->sat_cnt >= THRESHOLD_SATELLITE_CNT &&
      recognizer->sat_snr_sum >= THRESHOLD_SNR_SUM) {
    recognizer->status = OUTDOOR;
  } else {
    recognizer->status = INDOOR;
  }
}

bool mi_indoor_outdoor_recognizer_process(void *instance, uint64_t timestamp_ms,
                                          uint8_t *ids, uint8_t ids_len,
                                          uint8_t *snrs, uint8_t snrs_len) {
  MIIndoorOutdoorRecognizer *recognizer = (MIIndoorOutdoorRecognizer *)instance;

  // Initial update timestamp
  if (recognizer->update_timestamp_ms == 0) {
    recognizer->update_timestamp_ms = timestamp_ms;
  }

  // Update GPGSV status
  uint8_t id = 0;
  uint8_t snr = 0;
  ids_len = MIN(ids_len, SATELLITE_NUM_MAX);
  for (uint8_t i = 0; i < ids_len; ++i) {
    id = ids[i];
    snr = snrs[i];
    if (id > SATELLITE_NUM_MAX || id < 1) {
      // TODO: Maybe some exception happened
      continue;
    }
    update_gpgsv_info(recognizer, id, snr);
  }

  bool update = false;
  if (timestamp_ms - recognizer->update_timestamp_ms >
      RECOGNIZE_WIN_DURARION_MS) {
    update = true;
    reset_recognize_status(recognizer);
    recognizer->update_timestamp_ms = timestamp_ms;
    classify(recognizer);
  }

  return update;
}

bool mi_indoor_outdoor_recognizer_process_nmea_str(void *instance, char *str) {
  MIIndoorOutdoorRecognizer *recognizer = (MIIndoorOutdoorRecognizer *)instance;

  uint64_t ts = -1;
  uint8_t  ids_len = gpgsv_parse_nmea_str(str, &ts, recognizer->parse_ids,
                                         recognizer->parse_snrs,
                                         SINGLE_NEMA_SATELLITE_NUM_MAX);

  if (ids_len > 0) {
    return mi_indoor_outdoor_recognizer_process(
        recognizer, ts, recognizer->parse_ids, ids_len, recognizer->parse_snrs,
        ids_len);
  } else {
    return false;
  }
}

uint8_t mi_indoor_outdoor_recognizer_get_status(void *instance) {
  MIIndoorOutdoorRecognizer *recognizer = (MIIndoorOutdoorRecognizer *)instance;

  return recognizer->status;
}
