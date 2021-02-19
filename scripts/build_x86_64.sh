#!/bin/bash

./scripts/install_prerequisites.sh

set -e

export CFLAGS="-fPIC"
export CXXFLAGS="-fPIC -I${BUILD_DIR}/third_party/install/gtest/include"

OS='linux'
ARCH='x86_64'
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
      -DCMAKE_PREFIX_PATH=`pwd`/prerequisites/install/ \
      -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
      -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
      -DHOST=ON \
      -DMIOT_ALGO_ENABLE_TESTS=ON \
      . || exit 1

cmake --build ${BUILD_DIR} --target install || exit 1

# Package
cd ${BUILD_DIR}
make package
cd -

export PYTHONPATH=`pwd`/${BUILD_DIR}:$PYTHONPATH
python3 -c "import src.c.har_detector.har_detector"
python3 -c "import src.c.har_model.har_model"
