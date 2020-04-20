include_directories("${PROJECT_SOURCE_DIR}")

set(TARGET pypet_core)

# FIXME(Ying): This may lead to runtime error if users have multiple Python
# installed, and do not specify which Python to use when compiling.
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(PYPETCORE_SRCS "${PROJECT_SOURCE_DIR}/pypet/core/parser.cpp")

cc_library(
  ${TARGET}
  SHARED
  SRCS
  ${PYPETCORE_SRCS}
  DEPS
  torch
  isl)
