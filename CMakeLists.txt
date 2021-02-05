cmake_minimum_required(VERSION 3.12)
project(activity_recognition VERSION 0.1.0)

##--------------------------------------- Version info ---------------------------------------##
execute_process(COMMAND bash "-c" "cd ${CMAKE_CURRENT_SOURCE_DIR} && git rev-parse --short HEAD" OUTPUT_VARIABLE GIT_COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
string(TIMESTAMP BUILD_DATE "%Y-%m-%d %H:%M")
set(VERSION_INFO_STRING "xiaomi_miot_algo_har_version: ${PROJECT_VERSION}.${GIT_COMMIT_ID} (${BUILD_DATE})")
set(VERSION_NAME_STRING "MiotAlgoHAR-${PROJECT_VERSION}")
message(STATUS "Version info: ${VERSION_INFO_STRING}")
message(STATUS "Version name: ${VERSION_NAME_STRING}")

configure_file("${PROJECT_SOURCE_DIR}/src/c/miot_algo_har_version.c.in" "${PROJECT_SOURCE_DIR}/src/c/miot_algo_har_version.c")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
option(X86_32 "whether to build x86 32 bit library" OFF)
option(MIOT_ALGO_ENABLE_TESTS "whether to build c++ unit tests" OFF)

if (X86_32)
  message("Building configure for x86_32")
  add_compile_options(-m32)
endif(X86_32)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/c/)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/c/har_model)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/c/har_detector)

message(STATUS "MIOT_ALGO_ENABLE_TESTS = ${MIOT_ALGO_ENABLE_TESTS}")
if(MIOT_ALGO_ENABLE_TESTS)
  message("Enable unit test")

  include(${PROJECT_SOURCE_DIR}/third_party/third_party.cmake)
  add_subdirectory(test/cunit/har_model)
  add_subdirectory(test/cunit/har_detector)
endif(MIOT_ALGO_ENABLE_TESTS)
