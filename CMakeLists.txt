cmake_minimum_required(VERSION 3.10.2)
project(Deteccao)
set(source_dir "${PROJECT_SOURCE_DIR}")
file(GLOB source_files "${source_dir}/*.cpp")
find_package(OpenCV 3 REQUIRED)
find_package (Threads)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(Deteccao ${source_files})
target_link_libraries(Deteccao ${OpenCV_LIBS})
target_link_libraries(Deteccao ${CMAKE_THREAD_LIBS_INIT})

