set(TARGET pypet_core)

include_directories("${PROJECT_SOURCE_DIR}")

file(GLOB PYPETCORE_SRCS ${PROJECT_SOURCE_DIR}/pypet/core/*.cpp)

cc_library(
  ${TARGET}
  SHARED
  IMPORTED
  SRCS
  ${PYPETCORE_SRCS}
  DEPS
  pthread
  gflags
  glog
  isl
  torch
  torch_cpu
  c10
  python)
