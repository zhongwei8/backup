# Copyright (c) Xiaomi, 2020. All rights reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def get_mean_std(test_data):
    mean_val = np.mean(test_data)
    std_val = np.std(test_data)
    return mean_val, std_val


test_array = np.array([10, 9, 8, 36, 7, 77, 32]).astype(np.float32)
mean_val, std_val = get_mean_std(test_array)
print("mean val:{} std val:{}".format(mean_val, std_val))
# mean val:25.571428298950195 std val:23.801904678344727
