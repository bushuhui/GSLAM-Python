cmake_minimum_required(VERSION 2.8.12)
project(gslam)

add_subdirectory(pybind11)
pybind11_add_module(gslam cpp/gslam.cpp)
