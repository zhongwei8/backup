#!/bin/bash
#
# This script build Python binding for the C algorithm

./scripts/install_prerequisites.sh

set -e

BUILD_DIR=build_x86_64/
rm ${BUILD_DIR} -rf

cmake -B${BUILD_DIR} \
      -DCMAKE_PREFIX_PATH=`pwd`/prerequisites/install/ \
      -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
      -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
      -DHOST=ON \
      ./

cmake --build ${BUILD_DIR} --verbose

export PYTHONPATH=`pwd`/${BUILD_DIR}:$PYTHONPATH
python3 -c "import src.c.har_detector.har_detector"
python3 -c "import src.c.har_model.har_model"
