#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "indoor_outdoor_recognizer.c"


void output_gpgsv_info(MIIndoorOutdoorRecognizer *recognizer) {
    printf("timestamp_ms = %ld\n", recognizer->update_timestamp_ms);
    printf("snrs = ");
    for(uint8_t i = 0; i < 32; ++i) {
        printf("%d ", recognizer->snrs[i]);
    }
    printf("\n");
    printf("sat_cnt, sat_snr_sum, indoor_outdoor_status= %d, %d, %d\n",
        recognizer->sat_cnt,
        recognizer->sat_snr_sum,
        recognizer->status);
}

uint8_t main() {
    uint64_t timestamp = 1608730042147;
    uint8_t nums[4];
    uint8_t snrs[4];

    MIIndoorOutdoorRecognizer *recoginzer = mi_indoor_outdoor_recognizer_new();
    mi_indoor_outdoor_recognizer_init(recoginzer);

    uint64_t k = 1000;
    while(k) {

        timestamp += 10;
        uint8_t nums_len = 0;

        for(int _ = 0; _ < 4; ++_) {
            uint8_t num = rand() % 32;
            uint8_t snr = rand() % 100;
            if(rand() % 3 == 0) snr = 0;
            if(num < 0 || num > 31) continue;
            if(snr <= 0)    continue;
            nums[nums_len] = num;
            snrs[nums_len] = snr;
            nums_len++;
        }
        mi_indoor_outdoor_recognizer_process(recoginzer, timestamp, nums, snrs, nums_len, nums_len);
        output_gpgsv_info(recoginzer);
        k--;
    }

    return 0;
}
