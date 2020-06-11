set(TARGET pypet_core)

include_directories("${PROJECT_SOURCE_DIR}")

# TODO(Ying): Refine generating protobuf codes. Current implementations only
# work for that there is one *.proto in the source code directory.
set(PATH_PREFIX ${PROJECT_SOURCE_DIR}/pypoly/core/pypet)
file(GLOB PROTOBUF_FILE "${PATH_PREFIX}/*.proto")

get_filename_component(PROTO_PATH ${PROTOBUF_FILE} ABSOLUTE)
get_filename_component(PROTO_NAME ${PROTOBUF_FILE} NAME_WE)
protobuf_generate_cpp(PROTO_SRC PROTO_HDR "${PROTO_PATH}")
cc_library("${TARGET}_proto" SRCS "${PROTO_SRC}")
add_custom_command(
  TARGET "${TARGET}_proto"
  POST_BUILD
  COMMAND cp *.pb.* ${PATH_PREFIX}
  COMMENT "Copy generated C++ proto into directory pypoly/core/pypet."
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

py_proto_generate("${TARGET}_py_proto" SRCS ${PROTOBUF_FILE})
add_custom_command(
  TARGET "${TARGET}_py_proto"
  POST_BUILD
  COMMAND cp *.py ${PROJECT_SOURCE_DIR}/pypoly/python/
  COMMENT "Copy generated python proto into directory pypoly/python."
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB PYPETCORE_SRCS "${PATH_PREFIX}/*.cpp")
cc_library(
  ${TARGET}
  SHARED
  IMPORTED
  SRCS
  ${PYPETCORE_SRCS}
  DEPS
  ${TARGET}_proto
  pthread
  isl
  torch
  torch_cpu
  c10
  python
  protobuf)
