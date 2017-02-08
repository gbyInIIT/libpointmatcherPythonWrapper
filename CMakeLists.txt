cmake_minimum_required(VERSION 2.8.4)
project(libpointmatcherPythonWrapper)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(PYTHON_INC "/home/gao/anaconda2/include/python2.7")
set(PCL_INC "/usr/include/pcl-1.7")
set(EIGEN_INC "/usr/include/eigen3")

add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
include_directories(${PYTHON_INC})
include_directories(${PCL_INC})
include_directories(${EIGEN_INC})
set(SOURCE_FILES libpointmatcherPythonWrapper.cpp)
add_library(libpointmatcherPythonWrapper ${SOURCE_FILES})