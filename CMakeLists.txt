cmake_minimum_required(VERSION 3.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(PYPET_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PYPET_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PYPET_PYTHON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/build")

project(pypet CXX C)
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

include(third_party)
include(generic)
add_subdirectory(pypet)
