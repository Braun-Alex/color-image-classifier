cmake_minimum_required(VERSION 3.22.1)
project(color_image_classifier)

set(CMAKE_CXX_STANDARD 23)
set(HEADERS utils.h)
set(SOURCES utils.cpp main.cpp)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})
