include(ExternalProject)

set(PYBIND_PREFIX_DIR ${THIRD_PARTY_PATH}/pybind)
set(PYBIND_SOURCE_DIR ${THIRD_PARTY_PATH}/pybind/src/extern_pybind)
set(PYBIND_REPOSITORY https://github.com/pybind/pybind11.git)
set(PYBIND_TAG v2.2.4)

cache_third_party(
  extern_pybind
  REPOSITORY
  ${PYBIND_REPOSITORY}
  TAG
  ${PYBIND_TAG}
  DIR
  PYBIND_SOURCE_DIR)

set(PYBIND_INCLUDE_DIR ${PYBIND_SOURCE_DIR}/include)
include_directories(${PYBIND_INCLUDE_DIR})

ExternalProject_Add(
  extern_pybind
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${PYBIND_DOWNLOAD_CMD}"
  PREFIX ${PYBIND_PREFIX_DIR}
  SOURCE_DIR ${PYBIND_SOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
  set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/pybind_dummy.c)
  file(WRITE ${dummyfile} "const char * dummy_pybind = \"${dummyfile}\";")
  add_library(pybind STATIC ${dummyfile})
else()
  add_library(pybind INTERFACE)
endif()

add_dependencies(pybind extern_pybind)
