include(ExternalProject)

set(YAML_PREFIX_DIR ${THIRD_PARTY_PATH}/yaml)
set(YAML_SOURCE_DIR ${THIRD_PARTY_PATH}/yaml/src/extern_yaml)
set(YAML_INSTALL_DIR ${THIRD_PARTY_PATH}/yaml/src/extern_yaml-install)
set(YAML_LIBRARIES ${YAML_INSTALL_DIR}/lib)
set(YAML_REPOSITORY https://github.com/yaml/libyaml.git)
set(YAML_TAG dist-0.2.4)

cache_third_party(
  extern_yaml
  REPOSITORY
  ${YAML_REPOSITORY}
  TAG
  ${YAML_TAG}
  DIR
  YAML_SOURCE_DIR)

ExternalProject_Add(
  extern_yaml
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${YAML_DOWNLOAD_CMD}"
  PREFIX ${YAML_PREFIX_DIR}
  UPDATE_COMMAND ""
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${YAML_SOURCE_DIR}
  CONFIGURE_COMMAND autoreconf -f -i
  COMMAND ./configure --prefix=${YAML_INSTALL_DIR}
  BUILD_COMMAND $(MAKE) --silent -j $(nproc)
  INSTALL_COMMAND $(MAKE) install)
