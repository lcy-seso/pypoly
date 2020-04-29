set(TARGET pypet_core)

include_directories("${PROJECT_SOURCE_DIR}")

set(PYPETCORE_SRCS "${PROJECT_SOURCE_DIR}/pypet/core/parser.cpp")

cc_library(
  ${TARGET}
  SHARED
  IMPORTED
  SRCS
  ${PYPETCORE_SRCS}
  DEPS
  isl
  torch
  torch_cpu
  c10
  python)
