#!/bin/bash

set -e

BUILD_DIR=build_x86_32
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

cmake ./ \
  -B${BUILD_DIR} \
  -DX86_32=ON \
  $@ || exit 1

cmake --build ${BUILD_DIR} --verbose || exit 1
