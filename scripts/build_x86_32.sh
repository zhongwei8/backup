#!/bin/bash

set -e

export CFLAGS="-fPIC"
export CXXFLAGS="-fPIC -I${BUILD_DIR}/third_party/install/gtest/include"

OS='linux'
ARCH='x86_32'
ABI=${ARCH}
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
      -DX86_32=ON \
      . || exit 1

cmake --build ${BUILD_DIR} --target install || exit 1

# Package
cd ${BUILD_DIR}
make package
cd -
