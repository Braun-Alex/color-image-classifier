cmake_minimum_required(VERSION 3.22.1)
project(color_image_classifier)

set(CMAKE_CXX_STANDARD 23)
set(HEADERS utils.h phash.h color_moments.h core.h)
set(SOURCES utils.cpp phash.cpp color_moments.cpp core.cpp main.cpp)

find_package(CURL REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(${PROJECT_NAME} ${CURL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenCV_LIBS})