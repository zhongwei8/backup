// Copyright (c) Xiaomi. 2021. All rights reserved.
// Author: Farmer Li
// Date: 2021-02-23

#include "indoor_outdoor_recognizer.h"
#include "gtest/gtest.h"

TEST(MIIndoorOutdoorRecognizer, InitAndFree) {
  void* recognizer = mi_indoor_outdoor_recognizer_new();
  mi_indoor_outdoor_recognizer_init(recognizer);
  mi_indoor_outdoor_recognizer_free(recognizer);
}
