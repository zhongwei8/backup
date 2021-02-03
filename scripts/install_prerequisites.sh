#!/bin/bash
#
# Install third party depencies

set -e

# Download file and verify checksum
download_and_unzip() {
  PUSHD_DIR=`pwd`
  PROJECT_NAME=$1
  PROJECT_URL=$2
  PROJECT_SHA256SUM=$3
  ZIPFILE_NAME="${ROOT_DIR}/${PROJECT_NAME}.zip"

  cd "${ROOT_DIR}"
  echo "Downloading $URL as file ${ZIPFILE_NAME}"
  wget "${PROJECT_URL}" -O "${ZIPFILE_NAME}"
  echo "Verifying checksum of file $ZIPFILE_NAME"
  echo "${PROJECT_SHA256SUM} ${ZIPFILE_NAME}" | sha256sum --check
  echo "Unzipping ${ZIPFILE_NAME}"
  unzip -o ${ZIPFILE_NAME}
  cd "${PUSHD_DIR}"
}

project_build_dir() {
  PROJECT_NAME=$1
  ZIPFILE_NAME="${ROOT_DIR}/${PROJECT_NAME}.zip"
  UNZIP_DIR=`unzip -Z -1 "${ZIPFILE_NAME}" | cut -d"/" -f1 | head -n1`
  echo "${ROOT_DIR}/${UNZIP_DIR}"
}

# Build with cmake
cmake_build() {
  PUSHD_DIR=`pwd`
  PROJECT_NAME=$1

  cd `project_build_dir $PROJECT_NAME`
  shift
  cmake -Bcmake-build -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CXX_FLAGS="-fPIC" \
                -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT_DIR} \
                -DCMAKE_PREFIX_PATH=${INSTALL_ROOT_DIR} \
                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER} \
                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER} $@
  cmake --build cmake-build --verbose -j ${MAKE_JOBS}
  cd "${PUSHD_DIR}"
}

cmake_build_install() {
  PUSHD_DIR=`pwd`
  cmake_build $@

  PROJECT_NAME=$1
  cd `project_build_dir $PROJECT_NAME`
  cmake --build cmake-build --target install
  cd "${PUSHD_DIR}"
}

main() {
  mkdir -p prerequisites
  cd prerequisites
  export ROOT_DIR=`pwd`
  export INSTALL_ROOT_DIR=${ROOT_DIR}/install
  export MAKE_JOBS=8
  if command -v ccache; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
  fi


  PYBIND11_URL="https://cnbj1-inner-fds.api.xiaomi.net/miot-algorithm/tools/pybind11-v2.4.3.zip"
  PYBIND11_SHA256SUM=f1cc1e9c2836f9873aefdaf76a3280a55aae51068c759b27499a9cf34090d364
  download_and_unzip pybind11 $PYBIND11_URL $PYBIND11_SHA256SUM
  cmake_build_install pybind11 -DPYBIND11_TEST:BOOL=OFF
}

main "$@"
