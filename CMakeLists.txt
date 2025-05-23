cmake_minimum_required(VERSION 3.15)
project(BSplineX_Python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

set(BSPLINEX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/BSplineX")
set(BSPLINEX_SWIG_FILE "BSplineX.i")

add_subdirectory(${BSPLINEX_DIR})

set_property(
  SOURCE ${BSPLINEX_SWIG_FILE}
  PROPERTY CPLUSPLUS ON
)
set_property(
  SOURCE ${BSPLINEX_SWIG_FILE}
  PROPERTY USE_TARGET_INCLUDE_DIRECTORIES TRUE
)
set_property(
  SOURCE ${BSPLINEX_SWIG_FILE}
  PROPERTY SWIG_MODULE_NAME _bsplinex_impl
)

swig_add_library(libbsplinex
  TYPE MODULE
  LANGUAGE python
  SOURCES ${BSPLINEX_SWIG_FILE}
)

target_link_libraries(libbsplinex PRIVATE
  Python::Module
  BSplineX::BSplineX
)

if(MSVC)
  # Disable some warnings for Visual Studio.
  # Most of them are related to the SWIG generated code.
  target_compile_options(libbsplinex PRIVATE
    /wd4127 # conditional expression is constant
    /wd4456 # declaration of 'identifier' hides previous local declaration
    /wd4267 # conversion from 'size_t' to 'type', possible loss of data
  )
endif()

install(
  TARGETS libbsplinex
  DESTINATION bsplinex
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/_bsplinex_impl.py
  DESTINATION bsplinex
)

add_custom_command(TARGET libbsplinex POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_BINARY_DIR}/_bsplinex_impl.py
          ${CMAKE_SOURCE_DIR}/wrapper/bsplinex/_bsplinex_impl.py

  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:libbsplinex>
          ${CMAKE_SOURCE_DIR}/wrapper/bsplinex/$<TARGET_FILE_NAME:libbsplinex>
)

