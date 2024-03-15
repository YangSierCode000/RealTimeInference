cmake_minimum_required(VERSION 3.12)
project(RealSim)

# Add PyBind11
add_subdirectory(lib/pybind11)

# Include directories
include_directories(include/)

# Define the executable and link PyBind11
add_executable(MyExecutable cpp_src/main.cpp cpp_src/my_application.cpp)
target_link_libraries(MyExecutable PRIVATE pybind11::embed)
