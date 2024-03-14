cmake_minimum_required(VERSION 3.22.1)
project(color_image_classifier)

set(CMAKE_CXX_STANDARD 23)
set(HEADERS
        src/utils/utils.h
        src/cv_methods/color_moments/color_moments.h
        src/cv_methods/sift/sift.h
        src/metrics/metrics.h
        src/core/core.h
)
set(HEADER_PATHS
        src/utils
        src/cv_methods/color_moments
        src/cv_methods/sift
        src/metrics
        src/core
)
set(SOURCES
        src/utils/utils.cpp
        src/cv_methods/color_moments/color_moments.cpp
        src/cv_methods/sift/sift.cpp
        src/metrics/metrics.cpp
        src/core/core.cpp
        src/main.cpp
)

find_package(CURL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${HEADER_PATHS})
target_link_libraries(${PROJECT_NAME} ${CURL_LIBRARIES} OpenMP::OpenMP_CXX ${OpenCV_LIBS})