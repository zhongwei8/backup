#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-02-02

from pathlib import Path
import sys
import unittest

import pandas as pd

from test_utils import get_dataset

current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir / '../../src/py'))

from har_det import HarDetector
from har_det_c import HarDetector_C
from har_model import HarModel
from har_model_c import HarModel_C

DATASET_PATH = 'https://cnbj1.fds.api.xiaomi.com/new-sensor-bucket/sensor-data/activity-recognition/validation/MIOT-2020Q4-relabeled-test-validation.zip'
DATASET_NAME = 'MIOT-2020Q4-relabeled-test-validation'
DATASET_SHA256SUM = '559346cecc8914aa474bf5a3a497282d46ce48b1d904da16d9c9e4e7cd67c8e3'


class HarModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir: Path = get_dataset(DATASET_PATH,
                                          DATASET_NAME,
                                          sha256sum=DATASET_SHA256SUM)
        self.test_files = list(self.data_dir.rglob('*.csv'))
        self.detector = HarDetector()
        self.detector_c = HarDetector_C()
        self.model = HarModel()
        self.model_c = HarModel_C()
        return super().setUp()

    def test_stream_process(self):
        total = len(self.test_files)
        for i, file_path in enumerate(self.test_files, 1):
            print(f'\nTesting [{i:03d}/{total:03d}]')
            self.process_file(file_path)

    def process_file(self, file_path):
        print(f'Testing file: {file_path}')
        self.model.reset()
        self.model_c.reset()
        self.detector.reset()
        self.detector_c.reset()

        df = pd.read_csv(file_path)
        for i, row in df.iterrows():
            update = self.model.feed_data(row)
            update_c = self.model_c.feed_data(row)
            self.assertEqual(update, update_c)
            if update:
                res = self.model.get_result()
                res_c = self.model_c.get_result()
                self.assertEqual(res['Activity'], res_c['Activity'])
                self.assertEqual(res['Predict'], res_c['Predict'])
                self.assertAlmostEqual(res['Prob0'], res_c['Prob0'], places=5)
                self.assertAlmostEqual(res['Prob1'], res_c['Prob1'], places=5)
                self.assertAlmostEqual(res['Prob2'], res_c['Prob2'], places=5)
                self.assertAlmostEqual(res['Prob3'], res_c['Prob3'], places=5)
                self.assertAlmostEqual(res['Prob4'], res_c['Prob4'], places=5)
                self.assertAlmostEqual(res['Prob5'], res_c['Prob5'], places=5)

                # Test detector
                self.process_model_predict_result(res, res_c)

    def process_model_predict_result(self, result, result_c):
        self.detector.feed_data(result)
        self.detector_c.feed_data(result_c)
        res = self.detector.get_result()
        res_c = self.detector_c.get_result()
        self.assertEqual(res['PredictActivity'], res_c['PredictActivity'])


if __name__ == '__main__':
    unittest.main()
