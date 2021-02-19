include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src/c/)

set(HAR_TEST_LIBS mi_har gtest gtest_main)

add_executable(har_model_test har_model_test.cc)
target_link_libraries(har_model_test PUBLIC ${HAR_TEST_LIBS})
add_test(NAME har_model_test COMMAND har_model_test)

add_executable(har_detector_test har_detector_test.cc)
target_link_libraries(har_detector_test PUBLIC ${HAR_TEST_LIBS})
add_test(NAME har_detector_test COMMAND har_detector_test)

add_executable(activity_recognizer_test activity_recognizer_test.cc)
target_link_libraries(activity_recognizer_test PUBLIC ${HAR_TEST_LIBS})
add_test(NAME activity_recognizer_test COMMAND activity_recognizer_test)
