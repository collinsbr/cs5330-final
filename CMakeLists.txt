cmake_minimum_required(VERSION 3.0)
project(proj04)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE-DIRECTORIES})
add_executable(cam_cal src/cal_main.cpp include/calibration.h src/calibration.cpp include/csv_util.h src/csv_util.cpp)
add_executable(ar src/ar_main.cpp include/csv_util.h src/csv_util.cpp include/ar.h src/ar.cpp)
add_executable(gather src/gather_surfaces.cpp)
add_executable(match src/match_keypoints.cpp)

target_link_libraries(cam_cal ${OpenCV_LIBS})
target_link_libraries(ar ${OpenCV_LIBS})
target_link_libraries(gather ${OpenCV_LIBS})
target_link_libraries(match ${OpenCV_LIBS})
