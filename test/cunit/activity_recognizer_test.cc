// Copyright (c) Xiaomi. 2021. All rights reserved.
// Author: Farmer Li
// Date: 2021-02-19

#include "activity_recognizer.h"

#include "gtest/gtest.h"

TEST(MIActivityRecognizerTest, InitAndFree) {
  void* recognizer = mi_activity_recognizer_new();
  mi_activity_recognizer_init(recognizer, 0.8, 15);
  mi_activity_recognizer_free(recognizer);
}
