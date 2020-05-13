include(ExternalProject)

set(ISL_PREFIX_DIR ${THIRD_PARTY_PATH}/isl)
set(ISL_SOURCE_DIR ${THIRD_PARTY_PATH}/isl/src/extern_isl)
set(ISL_INSTALL_DIR ${THIRD_PARTY_PATH}/isl/src/extern_isl-install)
set(ISL_LIBRARIES ${ISL_INSTALL_DIR}/lib)
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

include_directories(${ISL_INSTALL_DIR}/include)
link_directories(${ISL_INSTALL_DIR}/lib)

ExternalProject_Add(
  extern_isl
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${ISL_DOWNLOAD_CMD}"
  PREFIX ${ISL_PREFIX_DIR}
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${ISL_SOURCE_DIR}
  CONFIGURE_COMMAND ./autogen.sh
  COMMAND ./configure --prefix=${ISL_INSTALL_DIR}
  BUILD_COMMAND $(MAKE) --silent -j $(nproc)
  INSTALL_COMMAND $(MAKE) install)
