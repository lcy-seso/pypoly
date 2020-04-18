include_directories(${TORCHLIB_INCLUDE_DIR})
link_directories(${TORCH_LIBRARIES})

include_directories("${PROJECT_SOURCE_DIR}")

set(TARGET pypet_core)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(PYPETCORE_SRCS "${PROJECT_SOURCE_DIR}/pypet/core/parser.cpp")

add_subdirectory(${PYBIND_SOURCE_DIR} build)
cc_library(${TARGET} SHARED SRCS ${PYPETCORE_SRCS} DEPS torch)
