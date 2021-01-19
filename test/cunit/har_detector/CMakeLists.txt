include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(../../../src/c/har_detector)
file(GLOB HAR_DETECTOR_CC_TEST_SRCS
  *.cc
  *.c
)


add_executable(har_detector_cc_test ${HAR_DETECTOR_CC_TEST_SRCS})
target_link_libraries(har_detector_cc_test PUBLIC
  har_detector_c
  gtest
  gtest_main
)

install(TARGETS har_detector_cc_test RUNTIME DESTINATION bin)
add_test(NAME har_detector_cc_test COMMAND har_detector_cc_test)
