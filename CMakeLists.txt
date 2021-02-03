cmake_minimum_required(VERSION 3.12)
project(activity_recognition VERSION 0.0.1)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/c/har_model)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/c/har_detector)
