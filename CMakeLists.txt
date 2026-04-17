cmake_minimum_required(VERSION 3.15)
project(QCQPsolver VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project includes Eigen headers from the bundled include/Eigen path
include_directories(${CMAKE_SOURCE_DIR}/include/Eigen)

add_executable(qcqp src/qcqp.cpp)

target_include_directories(qcqp PRIVATE ${CMAKE_SOURCE_DIR}/include/Eigen)
target_include_directories(qcqp PRIVATE ${CMAKE_SOURCE_DIR}/include/LBFGSpp)

if(MSVC)
    target_compile_options(qcqp PRIVATE /W4 /permissive-)
else()
    target_compile_options(qcqp PRIVATE -Wall -Wextra -Wpedantic)
endif()

message(STATUS "Building QCQPsolver executable")


