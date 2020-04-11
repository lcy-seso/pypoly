cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(ISL_EXP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ISL_EXP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

project(pypet CXX C)
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

include(third_party)

add_subdirectory(pypet)
