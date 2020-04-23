set(TEST_ROOT ${PROJECT_SOURCE_DIR}/pypet/core/tests)
set(TARGET pypet_test)

file(GLOB TEST_SRCS ${TEST_ROOT}/*.cpp)
include_directories(${GTEST_INCLUDE_DIR})

add_executable(${TARGET} ${TEST_SRCS})

target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/pypet/core)
target_link_libraries(${TARGET} PRIVATE torch gtest isl)
