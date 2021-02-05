/*
 * Copyright (c) Xiaomi. 2020. All rights reserved.
 * Description: This the har model unit test source file.
 *
 */
#include "har_model.h"

#include "gtest/gtest.h"


TEST(HarModelTest, DemoTest) {
  mi_har_model *inst = mi_har_model_new();
  mi_har_model_init(inst);
  mi_har_model_free(inst);
}
