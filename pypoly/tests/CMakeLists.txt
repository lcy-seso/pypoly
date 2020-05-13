set(TEST_ROOT ${PROJECT_SOURCE_DIR}/pypoly/tests)

cc_test_build(test1 SRCS ${TEST_ROOT}/test_pypet1.cpp DEPS pypet_core)

# TODO(Ying) still has bug.
# add_subdirectory(pet)
