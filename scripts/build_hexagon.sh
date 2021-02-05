#!/bin/bash

set -e

# HEXAGON_TOOLS is the path of "HEXAGON_Tools/6.*.*" toolchain
if [ -z "$HEXAGON_TOOLS" ]; then
  echo "HEXAGON_TOOLS is undefined";
  echo "Should be path to \"HEXAGON_Tools/6.*.*\""
fi

BUILD_DIR=build_hexagon/
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

cd ${BUILD_DIR}
cmake .. \
  -DHEXAGON_TOOLS=${HEXAGON_TOOLS} \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/hexagon6.toolchain.cmake \
  $@ || exit 1

cmake --build . -- -j || exit 1
cd -
