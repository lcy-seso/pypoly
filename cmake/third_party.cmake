set(THIRD_PARTY_PATH
    "${CMAKE_BINARY_DIR}/third_party"
    CACHE STRING
          "A path setting third party libraries download & build directories.")

set(THIRD_PARTY_CACHE_PATH
    "${CMAKE_SOURCE_DIR}"
    CACHE STRING
          "A path cache third party source code to avoid repeated download.")

set(THIRD_PARTY_BUILD_TYPE Release)

function(cache_third_party TARGET)
  set(options "")
  set(oneValueArgs URL REPOSITORY TAG DIR)
  set(multiValueArgs "")
  cmake_parse_arguments(cache_third_party "${optionps}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  string(REPLACE "extern_" "" TARGET_NAME ${TARGET})
  string(REGEX REPLACE "[0-9]+" "" TARGET_NAME ${TARGET_NAME})
  string(TOUPPER ${TARGET_NAME} TARGET_NAME)
  if(cache_third_party_REPOSITORY)
    set(${TARGET_NAME}_DOWNLOAD_CMD GIT_REPOSITORY
                                    ${cache_third_party_REPOSITORY})
    if(cache_third_party_TAG)
      list(APPEND ${TARGET_NAME}_DOWNLOAD_CMD GIT_TAG ${cache_third_party_TAG})
    endif()
  elseif(cache_third_party_URL)
    set(${TARGET_NAME}_DOWNLOAD_CMD URL ${cache_third_party_URL})
  else()
    message(
      FATAL_ERROR "Download link (Git repo or URL) must be specified for cache!"
    )
  endif()

  # Pass ${TARGET_NAME}_DOWNLOAD_CMD to parent scope, the double quotation marks
  # can't be removed
  set(${TARGET_NAME}_DOWNLOAD_CMD
      "${${TARGET_NAME}_DOWNLOAD_CMD}"
      PARENT_SCOPE)
endfunction()

include(external/isl)
include(external/pybind)
