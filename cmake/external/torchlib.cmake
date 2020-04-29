include(ExternalProject)

set(DEFAULT_PATH "${THIRD_PARTY_PATH}/torchlib/src")
set(TORCHLIB_PREFIX_DIR
    ${DEFAULT_PATH}
    CACHE
      STRING
      "Path of torch library. If not set, download torchlib from the Internet.")

if("${TORCHLIB_PREFIX_DIR}" STREQUAL ${DEFAULT_PATH})
  set(TORCHLIB_SOURCE_DIR ${TORCHLIB_PREFIX_DIR}/extern_torchlib)
  set(DOWNLOAD_TORCHLIB TRUE)
else()
  set(TORCHLIB_SOURCE_DIR ${TORCHLIB_PREFIX_DIR})
  set(DOWNLOAD_TORCHLIB FALSE)
endif()

message(STATUS "Use torch library : ${TORCHLIB_SOURCE_DIR}")

set(TORCH_LIBRARIES ${TORCHLIB_SOURCE_DIR}/lib)
set(TORCHLIB_INCLUDE_DIR ${TORCHLIB_SOURCE_DIR}/include)

include_directories(${TORCHLIB_INCLUDE_DIR})
link_directories(${TORCH_LIBRARIES})

if(${DOWNLOAD_TORCHLIB})
  set(TORCHLIB_REPOSITORY
      https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-
      latest.zip)
  set(TORCHLIB_TAG latest)

  cache_third_party(
    extern_torchlib
    URL
    ${TORCHLIB_REPOSITORY}
    TAG
    ${TORCHLIB_TAG}
    DIR
    TORCHLIB_SOURCE_DIR)

  ExternalProject_Add(
    externel_torchlib
    URL ${TORCHLIB_URL} ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
        "${TORCHLIB_DOWNLOAD_CMD}"
    PREFIX ${TORCHLIB_PREFIX_DIR}
    SOURCE_DIR ${TORCHLIB_SOURCE_DIR}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND "")
endif()

add_library(torchlib SHARED IMPORTED GLOBAL)
add_dependencies(torchlib extern_torchlib)
