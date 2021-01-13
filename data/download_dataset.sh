#!/bin/bash
set -e

work_dir=$(pwd)

mkdir -p sensor-data/har
cd sensor-data
echo $(pwd)
wget "https://cnbj1-fds.api.xiaomi.net/sensor-data/activity-recognition/cleaned/cleaned-0628-test.zip"
unzip -nq -d har/ "cleaned-0628-test.zip"

cd ${work_dir}