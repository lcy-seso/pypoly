include(ExternalProject)

set(PET_PREFIX_DIR ${THIRD_PARTY_PATH}/pet)
set(PET_SOURCE_DIR ${THIRD_PARTY_PATH}/pet/src/extern_pet)
set(PET_INSTALL_DIR ${THIRD_PARTY_PATH}/pet/src/extern_pet-install)
set(PET_LIBRARIES ${PET_INSTALL_DIR}/lib)
set(PET_REPOSITORY https://github.com/Meinersbur/pet.git)
set(PET_TAG pet-0.11.3)

cache_third_party(
  extern_pet
  REPOSITORY
  ${PET_REPOSITORY}
  TAG
  ${PET_TAG}
  DIR
  PET_SOURCE_DIR)

include_directories(${PET_INSTALL_DIR}/include)
link_directories(${PET_INSTALL_DIR}/lib)

ExternalProject_Add(
  extern_pet
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${PET_DOWNLOAD_CMD}"
  PREFIX ${PET_PREFIX_DIR}
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${PET_SOURCE_DIR}
  CONFIGURE_COMMAND ./autogen.sh
  COMMAND ./configure --prefix=${PET_INSTALL_DIR}
  BUILD_COMMAND $(MAKE) --silent -j $(nproc)
  INSTALL_COMMAND $(MAKE) install)
