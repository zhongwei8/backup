#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
#          Zhongwei Tian
# @Date: 2021-02-23

import numpy as np

RECOGNIZE_WIN_DURATION_MS = 4000
THRESHOLD_SATELLITE_CNT = 4
THRESHOLD_SNR_SUM = 80
SATELLITE_MAX = 32

UNDEFINED = 0
INDOOR = 1
OUTDOOR = 2


class MIIndoorOutdoorRecognizer():
    def __init__(self,
                 threshold_satellite_cnt=THRESHOLD_SATELLITE_CNT,
                 threshold_snr_sum=THRESHOLD_SNR_SUM,
                 win_duration_ms=RECOGNIZE_WIN_DURATION_MS):
        self.threshold_satellite_cnt = threshold_satellite_cnt
        self.threshold_snr_sum = threshold_snr_sum
        self.indoor_outdoor_width_ms = win_duration_ms

        self.begin_timestamp_ms = 0
        self.update_timestamp_ms = 0
        self.snrs = np.zeros(SATELLITE_MAX).astype(np.int)
        self.status = UNDEFINED
        self.satellite_num = 0
        self.satellite_snr_sum = 0

    def reset(self) -> None:
        self.begin_timestamp_ms = 0
        self.snrs.fill(0)
        self.status = UNDEFINED
        self.satellite_num = 0
        self.satellite_snr_sum = 0

    def _reset_recognize_status(self):
        self.snrs.fill(0)

    def _parse_gpgsv_sentence(self, gpgsv_sentence: str) -> tuple:

        assert (len(gpgsv_sentence) > 0)
        fields = gpgsv_sentence.split(',')
        field_cnt = len(fields)
        assert (field_cnt >= 1)

        timestamp_ms = fields[0]

        try:
            timestamp_ms = int(timestamp_ms)
        except ValueError:
            return (-1, [], [])

        if field_cnt <= 2 or fields[2] != '$GPGSV':
            return (timestamp_ms, [], [])

        if field_cnt <= 5:
            return (timestamp_ms, [], [])
        id_num = fields[5]
        if not id_num.isdigit():
            return (timestamp_ms, [], [])
        id_num = int(id_num)
        if id_num <= 0:
            return (timestamp_ms, [], [])

        field_idx = 6
        ids = []
        snrs = []
        while (True):
            if (field_idx >= field_cnt):
                break
            s_id = fields[field_idx]
            field_idx += 3
            if (field_idx >= field_cnt):
                break
            snr = fields[field_idx]
            field_idx += 1

            if (s_id == ''):
                continue
            if (snr == ''):
                snr = 0

            ids.append(int(s_id))
            snrs.append(int(snr))

        return timestamp_ms, ids, snrs

    def _update_gpgsv_info(self, num, snr) -> None:
        assert (1 <= num <= 32)
        num -= 1
        if (snr >= 0):
            self.snrs[num] = snr
        else:
            assert (snr >= 0)

    def _classify(self):
        print(f'Current satellite info arr: {self.snrs}')
        snr_cnt = 0
        snr_sum = 0
        for snr in self.snrs:
            if snr > 0:
                snr_cnt += 1
                snr_sum += snr

        if (snr_cnt >= THRESHOLD_SATELLITE_CNT
                and snr_sum >= THRESHOLD_SNR_SUM):
            self.status = OUTDOOR
        else:
            self.status = INDOOR

        self.satellite_num = snr_cnt
        self.satellite_snr_sum = snr_sum
        print(
            f'Classify: {self.status}, {self.satellite_num}, {self.satellite_snr_sum}\n'
        )

    def process(self, timestamp_ms: int, ids: list, snrs: list) -> None:
        if (self.begin_timestamp_ms == 0):
            self.begin_timestamp_ms = timestamp_ms
        ids_len, snrs_len = len(ids), len(snrs)
        assert (ids_len == snrs_len)
        assert (ids_len <= SATELLITE_MAX)

        for i in range(ids_len):
            num, snr = ids[i], snrs[i]
            self._update_gpgsv_info(num, snr)

        update = False
        if (timestamp_ms - self.update_timestamp_ms >
                RECOGNIZE_WIN_DURATION_MS):
            update = True
            self._classify()
            self._reset_recognize_status()
            self.update_timestamp_ms = timestamp_ms

        return update

    def process_with_raw_nmea_sentence(self, nmea_sentence: str) -> None:
        timestamp_ms, nums, snrs = self._parse_gpgsv_sentence(nmea_sentence)
        # print(f'timestamp_ms, nums, snrs = {timestamp_ms}, {nums}, {snrs}')
        if timestamp_ms == -1:
            return False
        else:
            return self.process(timestamp_ms, nums, snrs)

    def get_status(self) -> int:
        return self.status

    def get_satellite_status(self):
        return self.satellite_num, self.satellite_snr_sum

    def get_update_timestamp(self):
        return self.update_timestamp_ms
