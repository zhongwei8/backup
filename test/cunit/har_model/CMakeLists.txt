include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(../../../src/c/har_model)

file(GLOB HAR_MODEL_CC_TEST_SRCS
  *.cc
  *.c
)


add_executable(har_model_cc_test ${HAR_MODEL_CC_TEST_SRCS})
target_link_libraries(har_model_cc_test PUBLIC
  har_model_c
  gtest
  gtest_main
)

install(TARGETS har_model_cc_test RUNTIME DESTINATION bin)

add_test(NAME har_model_cc_test COMMAND har_model_cc_test)
