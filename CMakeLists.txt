cmake_minimum_required(VERSION 3.12)
project(activity_recognition VERSION 0.1.0)

option(X86_32 "whether to build x86 32 bit library" OFF)
option(MIOT_ALGO_ENABLE_TESTS "whether to build c++ unit tests" OFF)
if (NOT DEFINED OS)
  message(STATUS "OS not set, set to default: linux")
  set(OS "linux")
endif ()
if (NOT DEFINED ARCH)
  message(STATUS "ARCH not set, set to default: x86_64")
  set(ARCH "x86_64")
endif ()
if (OS STREQUAL "linux" AND ARCH STREQUAL "x86_64")
  set(HOST ON)
  set(MIOT_ALGO_ENABLE_TESTS ON)
endif()

##---------------------- Version info ----------------------------------------##
execute_process(COMMAND bash "-c" "cd ${CMAKE_CURRENT_SOURCE_DIR} && git rev-parse --short HEAD" OUTPUT_VARIABLE GIT_COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
string(TIMESTAMP BUILD_DATE "%Y-%m-%d %H:%M")
set(VERSION_INFO_STRING "xiaomi_miot_algo_har_version: ${PROJECT_VERSION}.${GIT_COMMIT_ID} (${BUILD_DATE})")
set(VERSION_NAME_STRING "MiHAR-${PROJECT_VERSION}")
message(STATUS "Version info: ${VERSION_INFO_STRING}")
message(STATUS "Version name: ${VERSION_NAME_STRING}")

configure_file("${PROJECT_SOURCE_DIR}/src/c/miot_algo_har_version.c.in" "${PROJECT_SOURCE_DIR}/src/c/miot_algo_har_version.c")

##---------------------- Compile Options -------------------------------------##
add_compile_options(-g -Wall)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if (X86_32)
  message("Building configure for x86_32")
  add_compile_options(-m32)
endif(X86_32)

##---------------------- Include Paths ---------------------------------------##
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/c/)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/c/har_model)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/c/har_detector)

set(MI_HAR_LIB_C
  src/c/activity_recognizer.c
  src/c/miot_algo_har_version.c
  src/c/har_detector/har_detector.c
  src/c/har_model/har_model.c
)

add_library(mi_har_pure ${MI_HAR_LIB_C})
set_target_properties(mi_har_pure PROPERTIES PUBLIC_HEADER  src/c/activity_recognizer.h)

##---------------------- Install and Package ---------------------------------##
set(CPACK_GENERATOR "ZIP")
set(CPACK_SET_DESTDIR ON)
set(CPACK_INSTALL_PREFIX "")
set(CUSTOM_SYSTEM_NAME ${OS})
if (ARCH)
  set(CUSTOM_SYSTEM_NAME ${OS}-${ARCH})
endif()
set(HAR_PACKAGE_NAME MiHAR-${PROJECT_VERSION}-${CUSTOM_SYSTEM_NAME})
set(CPACK_PACKAGE_FILE_NAME ${HAR_PACKAGE_NAME})
set(CPACK_OUTPUT_FILE_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/${HAR_PACKAGE_NAME})

if ((OS STREQUAL "linux") OR (OS STREQUAL "generic" AND ARCH STREQUAL "qdsp"))
  set(HAR_MODEL_EXTRA_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/${ARCH}/)
  find_library(LIB_HAR_CNN model_har_cnn HINTS ${HAR_MODEL_EXTRA_LIB_PATH})
  find_library(LIB_MICRO micro HINTS ${HAR_MODEL_EXTRA_LIB_PATH})
  target_link_libraries(mi_har ${LIB_HAR_CNN} ${LIB_MICRO})
  install(FILES ${LIB_HAR_CNN} TYPE LIB)
  install(FILES ${LIB_MICRO} TYPE LIB)
endif()

install(TARGETS mi_har
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)

# Host Only
if(HOST)
  set_target_properties(mi_har PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  message("Build activity recognizer for python interface")
  find_package(PythonLibs 3.5 REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})
  find_package(pybind11)
  pybind11_add_module(mi_har_py src/c/activity_recognizer_py.cc)
  set_target_properties(mi_har_py PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  target_link_libraries(mi_har_py PRIVATE mi_har)
  install(TARGETS mi_har_py LIBRARY DESTINATION lib)
endif()

# Unit Test
if(MIOT_ALGO_ENABLE_TESTS)
  message(STATUS "Enable unit test")

  include(${PROJECT_SOURCE_DIR}/third_party/third_party.cmake)
  add_subdirectory(test/cunit)
endif(MIOT_ALGO_ENABLE_TESTS)

include(CPack)
