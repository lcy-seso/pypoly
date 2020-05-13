set(TEST_ROOT ${PROJECT_SOURCE_DIR}/pypoly/tests)

cc_test_build(test1 SRCS ${TEST_ROOT}/test_pypet1.cpp DEPS pypet_core)

include_directories(${PET_SOURCE_DIR} ${PET_INSTALL_DIR}/include
                    ${YAML_INSTALL_DIR}/include)
link_directories(${PET_INSTALL_DIR}/lib ${YAML_INSTALL_DIR}/lib)
cc_test_build(
  test_pet
  SRCS
  ${TEST_ROOT}/test_pet.cpp
  ${PET_SOURCE_DIR}/emit.c
  DEPS
  pet
  yaml
  isl)
