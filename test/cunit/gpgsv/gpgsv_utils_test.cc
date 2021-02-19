// Copyright (c) Xiaomi. 2021. All rights reserved.
// Author: Farmer Li
// Date: 2021-02-23

#include "gpgsv/gpgsv_utils.h"

#include "gtest/gtest.h"

TEST(MIActivityRecognizerTest, gpgsv_read_one_interge_1) {
  char* str = "26,ABCDEFGHILK";

  uint8_t val = 0;
  char*   pc = gpgsv_read_one_interge(str, &val);

  EXPECT_EQ(val, 26);
  EXPECT_EQ(*pc, 'A');
}

TEST(MIActivityRecognizerTest, gpgsv_read_one_interge_2) {
  char* str = "26";

  uint8_t val = 0;
  char*   pc = gpgsv_read_one_interge(str, &val);

  EXPECT_EQ(val, 26);
  EXPECT_EQ(*pc, '\0');
}

TEST(MIActivityRecognizerTest, gpgsv_read_one_interge_3) {
  char* str = "1*54";

  uint8_t val = 0;
  char*   pc = gpgsv_read_one_interge(str, &val);

  EXPECT_EQ(val, 1);
  EXPECT_EQ(*pc, '5');
}

TEST(MIActivityRecognizerTest, gpgsv_get_one_satellite_id_and_snr_1) {
  char* str = "26,00,000,40,1*55";

  uint8_t id = 0;
  uint8_t snr = 0;
  char*   pc = gpgsv_get_one_satellite_id_and_snr(str, &id, &snr);

  EXPECT_EQ(id, 26);
  EXPECT_EQ(snr, 40);
  EXPECT_EQ(*pc, '1');
}

TEST(MIActivityRecognizerTest, gpgsv_get_one_satellite_id_and_snr_2) {
  char* str = "26,00,000,40";

  uint8_t id = 0;
  uint8_t snr = 0;
  char*   pc = gpgsv_get_one_satellite_id_and_snr(str, &id, &snr);

  EXPECT_EQ(id, 26);
  EXPECT_EQ(snr, 40);
  EXPECT_EQ(*pc, '\0');
}

TEST(MIActivityRecognizerTest, gpgsv_get_one_satellite_id_and_snr_3) {
  char* str = "1*54";

  uint8_t id = 0;
  uint8_t snr = 0;
  char*   pc = gpgsv_get_one_satellite_id_and_snr(str, &id, &snr);

  EXPECT_EQ(id, 1);
  EXPECT_EQ(snr, 255);
  EXPECT_TRUE(pc == NULL);
}

TEST(MIActivityRecognizerTest, gpgsv_parse_nmea_str) {
  char* str =
      "1607938224367,1607938224363,$GPGSV,1,1,03,16,00,000,26,26,00,000,32,27,"
      "00,000,22,1*54";

  uint8_t ids_len = 0;
  uint64_t ts = 0;
  uint8_t ids[4];
  uint8_t snrs[4];
  uint8_t ids_ex[] = {16, 26, 27};
  uint8_t snrs_ex[] = {26, 32, 22};
  ids_len = gpgsv_parse_nmea_str(str, &ts, ids, snrs, 4);

  EXPECT_EQ(ts, 1607938224367);
  EXPECT_EQ(ids_len, sizeof(ids_ex) / sizeof(ids_ex[0]));
  for (int i = 0; i < ids_len; i++) {
    EXPECT_EQ(ids[i], ids_ex[i]);
    EXPECT_EQ(snrs[i], snrs_ex[i]);
  }
}
