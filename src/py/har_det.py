# Copyright (c) Xiaomi, 2020. All rights reserved.
from __future__ import absolute_import, division, print_function

from enum import Enum, unique
from pathlib import Path
import sys

import numpy as np

cur_dir = Path(__file__).parent.resolve()
root_dir = cur_dir / '../../../../../ai-algorithm-depolyment'
if root_dir.exists():
    sys.path.append(root_dir)
else:
    root_dir = cur_dir / '../../../ai-algorithm-depolyment'
    if root_dir.exists():
        sys.path.append(root_dir)

# Import from ai-algorithm-depolyment repo
# So this file dependent on ai-algorithm-depolyment
from utils.base import SensorAlgo


@unique
class ActivityType(Enum):
    Unknown = 0
    Walk = 1
    Run = 2
    Rowing = 3
    Elliptical = 4
    Bike = 5
    NumType = 6


@unique
class HarStateMachine(Enum):
    Idle = 0
    Prepare = 1
    PrepareBreak = 2
    Stable = 3
    StableBreak = 4


TimeThd = [0, 600, 180, 180, 180, 600]


class HarDetector(SensorAlgo):
    def __init__(self,
                 buf_len=30,
                 vote_len=15,
                 num_classes=6,
                 vote_score_thd=0.8):
        self._vote_score_thd = vote_score_thd
        self._buf_len = buf_len
        self._buffer = np.zeros(self._buf_len, dtype=int)
        self._cnt = 0
        self._idx = 0
        self._activity = 0
        self._version = 1000001
        self._vote_len = vote_len
        self._num_classes = num_classes
        self._input_names = [
            'EventTimestamp(ns)', 'Prob0', 'Prob1', 'Prob2', 'Prob3', 'Prob4',
            'Prob5', 'Activity'
        ]
        self._output_names = [
            'EventTimestamp(ns)', 'Activity', 'PredictActivity'
        ]
        self._algo_name = 'HarDetector'
        self._cur_timestamp = 0
        self._res = {}

    def reset(self):
        self._buffer = np.zeros(self._buf_len, dtype=int)
        self._cnt = 0
        self._idx = 0
        self._cur_timestamp = 0
        self._activity = 0
        self._res = {}

    def is_realtime(self):
        return True

    def argmax(self, probs):
        p_max = 0
        max_idx = 0
        for i in range(self._num_classes):
            if p_max < probs[i]:
                p_max = probs[i]
                max_idx = i
        return max_idx

    def update_predict_buffer(self, probs, idx):
        cur_predict = self.argmax(probs)
        self._buffer[idx] = cur_predict
        return cur_predict

    def vote_majority(self, vote_len):
        votes = np.zeros(self._num_classes)
        for n in range(vote_len):
            i = self._idx - n
            if i < 0:
                i = self._buf_len + i
            d = self._buffer[i]
            votes[d] += 1
        candidate = np.argmax(votes)
        vote_score = votes[candidate] / vote_len
        return candidate, vote_score

    def process(self, probs):
        cur_predict = self.update_predict_buffer(probs, self._idx)
        if self._cnt >= self._vote_len:
            cur_predict, score = self.vote_majority(self._vote_len)
            if score >= self._vote_score_thd:
                self._activity = cur_predict
        else:
            self._activity = cur_predict

    def feed_data(self, data_point):
        probs = np.zeros(self._num_classes)
        self._cur_timestamp = data_point['EventTimestamp(ns)']
        probs[0] = data_point['Prob0']
        probs[1] = data_point['Prob1']
        probs[2] = data_point['Prob2']
        probs[3] = data_point['Prob3']
        probs[4] = data_point['Prob4']
        probs[5] = data_point['Prob5']
        self.process(probs)
        self._cnt += 1
        self._idx += 1
        if self._idx >= self._buf_len:
            self._idx = 0
        self._res = {
            'EventTimestamp(ns)': self._cur_timestamp,
            'Activity': data_point['Activity'],
            'PredictActivity': self._activity,
        }
        return True

    def get_result(self):
        return self._res


class HarDetectorFSM(HarDetector):
    def __init__(self, buf_len=30, vote_len=10, num_classes=6, threds=None):
        self._buf_len = buf_len
        self._buffer = np.zeros(self._buf_len, dtype=int)
        self._cnt = 0
        self._idx = 0
        self._activity = ActivityType.Unknown.value
        self._state = HarStateMachine.Idle.value
        self._version = 1000001
        self._vote_len = vote_len
        self._num_classes = num_classes
        self._start_prepare_ts = 0
        self._start_break_ts = 0
        self._input_names = [
            'EventTimestamp(ns)', 'Prob0', 'Prob1', 'Prob2', 'Prob3', 'Prob4',
            'Prob5'
        ]
        self._output_names = ['EventTimestamp(ns)', 'PredictActivity']
        self._algo_name = 'HarDetectorFSM'
        self._cur_timestamp = 0
        if threds is None or len(threds) < num_classes:
            self._threds = [0, 0, 0, 0, 0, 0]
        elif len(threds) == num_classes:
            self._threds = threds
        else:
            raise Exception("invalid input threds, length of threds {0}"
                            "should be equal to num_classes: {1}.".format(
                                len(threds), self._num_classes))

    def reset(self):
        self._buffer = np.zeros(self._buf_len, dtype=int)
        self._cnt = 0
        self._idx = 0
        self._cur_timestamp = 0
        self._activity = ActivityType.Unknown.value
        self._state = HarStateMachine.Idle.value
        self._start_prepare_ts = 0
        self._start_break_ts = 0

    def vote_candidate(self, candidate, vote_len):
        vote = 0
        for n in range(vote_len):
            i = self._idx - n
            if i < 0:
                i = self._buf_len + i
            if self._buffer[i] == candidate:
                vote += 1
        return vote / vote_len

    def process_idle_state(self):
        cur_predict, vote_score = self.vote_majority(self._vote_len)
        if cur_predict != ActivityType.Unknown.value and vote_score >= 0.8:
            self._activity = cur_predict
            self._state = HarStateMachine.Prepare.value
            self._start_prepare_ts = self._cur_timestamp
        else:
            self._activity = ActivityType.Unknown.value
            self._state = HarStateMachine.Idle.value

    def process_prepare_state(self):
        vote_score = self.vote_candidate(self._activity, self._vote_len)
        if vote_score < 0.8:
            self._state = HarStateMachine.PrepareBreak.value
            self._start_break_ts = self._cur_timestamp
        elif self._cur_timestamp - self._start_prepare_ts >= TimeThd[
                self._activity] * 1000:
            self._state = HarStateMachine.Stable.value

    def process_preparebreak_state(self):
        vote_score = self.vote_candidate(self._activity, self._vote_len)
        if vote_score >= 0.8:
            self._state = HarStateMachine.Prepare.value
        else:
            cur_predict, vote_score = self.vote_majority(self._vote_len)
            if vote_score >= 0.8:
                self._state = HarStateMachine.Prepare.value
                self._activity = cur_predict
                self._start_prepare_ts = self._cur_timestamp
            elif self._cur_timestamp - self._start_prepare_ts >= 30 * 1000:
                self._state = HarStateMachine.Idle.value
                self._activity = ActivityType.Unknown.value

    def process_stable_state(self):
        vote_score = self.vote_candidate(self._activity, self._vote_len)
        if vote_score < 0.8:
            self._state = HarStateMachine.StableBreak.value
            self._start_break_ts = self._cur_timestamp

    def process_stablebreak_state(self):
        vote_score = self.vote_candidate(self._activity, self._vote_len)
        if vote_score >= 0.8:
            self._state = HarStateMachine.Stable.value
        else:
            cur_predict, vote_score = self.vote_majority(self._vote_len)
            if vote_score >= 0.8 and cur_predict != ActivityType.Unknown.value:
                self._state = HarStateMachine.Prepare.value
                self._activity = cur_predict
                self._start_prepare_ts = self._cur_timestamp
            elif self._cur_timestamp - self._start_break_ts >= 60 * 1000:
                self._state = HarStateMachine.Idle.value
                self._activity = ActivityType.Unknown.value

    def process_state_machine(self):
        if self._state == HarStateMachine.Idle.value:
            self.process_idle_state()
        elif self._state == HarStateMachine.Prepare.value:
            self.process_prepare_state()
        elif self._state == HarStateMachine.PrepareBreak.value:
            self.process_preparebreak_state()
        elif self._state == HarStateMachine.Stable.value:
            self.process_stable_state()
        elif self._state == HarStateMachine.StableBreak.value:
            self.process_stablebreak_state()
        else:
            raise Exception("Unknown state machine state {0}".format(
                self._state))

    def process(self, probs):
        self.update_predict_buffer(probs, self._idx)
        self.process_state_machine()
