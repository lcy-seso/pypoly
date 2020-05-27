# FIXME(Ying): This may lead to runtime error if users have multiple Python
# installed, and do not specify which Python to use when compiling.
find_package(PythonLibs REQUIRED)
message(STATUS "Python include dir: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Python library: ${PYTHON_LIBRARY}")

add_library(python SHARED IMPORTED GLOBAL)
set_property(TARGET python PROPERTY IMPORTED_LOCATION ${PYTHON_LIBRARIES})
include_directories(${PYTHON_INCLUDE_DIRS})
