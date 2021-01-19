/*
 * Copyright (c) Xiaomi. 2020. All rights reserved.
 * Description: This the har detector algorithm source file.
 *
 */
#include "gtest/gtest.h"
#include "har_detector_func.h"

TEST(HarDetectorTest, Argmax) {
  float   test_data[6] = {0.1, 0.15, 0.3, 0.2, 0.02, 0.23};
  uint8_t expected_result = 2;
  uint8_t ret = argmax_float(test_data, 6);

  EXPECT_EQ(expected_result, ret);
}

TEST(HarDetectorTest, VoteMajorityScore) {
  uint8_t test_data[15] = {1, 2, 3, 4, 5, 2, 5, 3, 3, 0, 0, 3, 4, 5, 3};
  vote_candidate_s vote_result, expected_result;
  expected_result.type = 3;
  expected_result.score = 0.4;
  vote_majority_score(test_data, 11, 10, &vote_result);
  EXPECT_EQ(expected_result.type, vote_result.type);
  EXPECT_EQ(expected_result.score, vote_result.score);

  expected_result.type = 3;
  expected_result.score = 0.3;
  vote_majority_score(test_data, 12, 10, &vote_result);
  EXPECT_EQ(expected_result.type, vote_result.type);
  EXPECT_EQ(expected_result.score, vote_result.score);

  expected_result.type = 3;
  expected_result.score = 0.6;
  vote_majority_score(test_data, 11, 5, &vote_result);
  EXPECT_EQ(expected_result.type, vote_result.type);
  EXPECT_EQ(expected_result.score, vote_result.score);
}
