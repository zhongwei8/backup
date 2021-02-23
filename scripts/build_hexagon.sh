#!/bin/bash

set -e

# HEXAGON_TOOLS is the path of "HEXAGON_Tools/6.*.*" toolchain
if [ -z "$HEXAGON_TOOLS" ]; then
  echo "HEXAGON_TOOLS is undefined";
  echo "Should be path to \"HEXAGON_Tools/6.*.*\""
  exit 1
fi

OS='generic'
ARCH='qdsp'
CUSTOM_SYSTEM_NAME="${OS}-${ARCH}"
BUILD_DIR="build_${CUSTOM_SYSTEM_NAME}"/
INSTALL_DIR="${BUILD_DIR}/output/"

# Clear
echo "Clean build directory: ${BUILD_DIR}"
rm -rf ${BUILD_DIR}

cmake -B${BUILD_DIR} \
      -DOS=$OS \
      -DARCH=$ARCH \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -DHEXAGON_TOOLS=${HEXAGON_TOOLS} \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/hexagon6.toolchain.cmake \
      . || exit 1

cmake --build ${BUILD_DIR} --target install -- -j || exit 1

# Package
cd ${BUILD_DIR}
make package
cd -
