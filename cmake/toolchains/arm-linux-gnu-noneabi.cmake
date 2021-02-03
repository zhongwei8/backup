set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER "${CROSS_COMPILE_TOOLS}/bin/arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "${CROSS_COMPILE_TOOLS}/bin/arm-none-eabi-g++")
set(CMAKE_AR "${CROSS_COMPILE_TOOLS}/bin/arm-none-eabi-gcc-ar")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(ARM ON)

add_compile_options(-mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -ffunction-sections -fdata-sections -g -Os -MMD -MP -std=c99 -Wall -s)
