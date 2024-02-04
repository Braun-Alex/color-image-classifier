cmake_minimum_required(VERSION 3.22.1)
project(color_image_classifier)

set(CMAKE_CXX_STANDARD 23)
set(HEADERS utils.h color_moments.h)
set(SOURCES utils.cpp color_moments.cpp main.cpp)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})