function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  if(cc_library_SRCS)
    if(cc_library_SHARED) # build *.so
      add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
    endif()

    if(cc_library_DEPS)
      # remove link to python, see notes at: https://github.com/pybind/pybind11/
      # blob/master/docs/compiling.rst#building-manually
      if("${cc_library_DEPS};" MATCHES "python;")
        list(REMOVE_ITEM cc_library_DEPS python)
        add_dependencies(${TARGET_NAME} python)
        target_link_libraries(${TARGET_NAME} "-Wl,-undefined,dynamic_lookup")
      endif()
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
    endif()

    # cpplint code style
    foreach(source_file ${cc_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cc_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()
  else(cc_library_SRCS)
    if(cc_library_DEPS)
      list(REMOVE_DUPLICATES cc_library_DEPS)
      set(target_SRCS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c)
      file(WRITE ${target_SRCS}
           "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")
      add_library(${TARGET_NAME} STATIC ${target_SRCS})
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL_ERROR "No source file is given.")
    endif()
  endif(cc_library_SRCS)
endfunction(cc_library)
