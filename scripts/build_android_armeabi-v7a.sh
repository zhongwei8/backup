#!/bin/bash

set -e

# ANDROID_NDK is the path of "xxx/ndk/22.0.7026061/"
if [ -z "$ANDROID_NDK" ]; then
  echo "ANDROID_NDK is undefined";
  echo "Should be path of Android NDK, like \"xxx/ndk/22.0.7026061/\""
  exit 1
fi

OS='android'
ARCH='armeabi-v7a'
ABI=${ARCH}
MINSDKVERSION=28
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
      -DCPACK_SYSTEM_NAME=${CUSTOM_SYSTEM_NAME} \
      -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=$ABI \
      -DANDROID_ARM_MODE='arm' \
      -DANDROID_ARM_NEON='FALSE' \
      -DANDROID_NATIVE_API_LEVEL=$MINSDKVERSION \
      . || exit 1

cmake --build ${BUILD_DIR} --target install -- -j || exit 1

# Package
cd ${BUILD_DIR}
make package
cd -
