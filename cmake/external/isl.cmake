include(ExternalProject)

set(ISL_PREFIX_DIR ${THIRD_PARTY_PATH}/isl)
set(ISL_SOURCE_DIR ${THIRD_PARTY_PATH}/isl/src/extern_isl)
set(ISL_INSTALL_DIR ${THIRD_PARTY_PATH}/isl/src/extern_isl-install)
set(ISL_REPOSITORY https://github.com/Meinersbur/isl.git)
set(ISL_TAG isl-0.22)

cache_third_party(
  extern_isl
  REPOSITORY
  ${ISL_REPOSITORY}
  TAG
  ${ISL_TAG}
  DIR
  ISL_SOURCE_DIR)

set(ISL_INCLUDE_DIR ${ISL_SOURCE_DIR}/include)
include_directories(${ISL_INCLUDE_DIR})

ExternalProject_Add(
  extern_isl
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${ISL_DOWNLOAD_CMD}"
  PREFIX ${ISL_PREFIX_DIR}
  SOURCE_DIR ${ISL_SOURCE_DIR}
  UPDATE_COMMAND ""
  COMMAND ${ISL_SOURCE_DIR}/./autogen.sh
  CONFIGURE_COMMAND ${ISL_SOURCE_DIR}/./configure --prefix=${ISL_INSTALL_DIR}
  BUILD_COMMAND make -j $(nproc)
  INSTALL_COMMAND make install
  TEST_COMMAND "")

add_library(isl INTERFACE)

add_dependencies(isl extern_isl)
