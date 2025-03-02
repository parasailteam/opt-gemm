cmake_minimum_required(VERSION 3.22)

project(OptGemm
  VERSION 0.0.0
  LANGUAGES CXX)

set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(CheckLanguage)

#### Options ####
option(ENABLE_CUDA "Enable CUDA"                    ON)
option(ENABLE_HIP  "Enable ROCM/HIP"               OFF)
option(PYMODULE    "Install and Build as PyModule" OFF)

if (PYMODULE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
else()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0")
    message (FATAL_ERROR "Require Clang 12.0 or later")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
    message (FATAL_ERROR "Require GCC/G++ 11.0 or later")
  endif()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
  set(WARNING_FLAGS "-Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter")
endif()

add_compile_definitions (OPT_GEMM_VERSION=${CMAKE_PROJECT_VERSION})

if(ENABLE_CUDA)
  check_language(CUDA OPTIONAL)
  if (CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "80")
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    add_definitions(-DENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 ${WARNING_FLAGS}")
    if (CMAKE_BUILD_TYPE MATCHES "Debug")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G ${WARNING_FLAGS}")
    endif()
    add_compile_definitions(OPT_GEMM_CUDA_VERSION=${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})
  else()
     set(ENABLE_CUDA OFF)
     message(ERROR "Cannot find CUDA. Disabling CUDA backend.")
  endif()
endif()

# TODO set these as RELEASE flags

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -march=x86-64-v2 -O3 ${WARNING_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -march=x86-64-v2 ${WARNING_FLAGS}")

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CUDA ${SRC}/cuda)

set(KERNEL_HEADER_FILES)
set(GEMM_LIB_DIR)
set(CUDA_SOURCES)

if(ENABLE_CUDA)
  if (FULL_TUNE)
  else()
    set(FK_CUDA_ARCH "")
    foreach(SM_ARCH ${CMAKE_CUDA_ARCHITECTURES})
      if ((SM_ARCH STREQUAL "80") OR (SM_ARCH STRGREATER "80"))
        set(arch "ampere")
      elseif (SM_ARCH STREQUAL "70")
        set(arch "volta")
      elseif (SM_ARCH STRLESS 70)
        set(arch "maxwell")
      endif()
      set(FK_CUDA_ARCH ${arch} ${FK_CUDA_ARCH})
    endforeach()

    execute_process(COMMAND python3 ${SRC}/gen_kernels.py)
  endif()

  set(GEMM_LIB_DIR "cutlass/include")
  include(${CUDA}/ampere/kernels/kernels.cmake)
  
  if (EXISTS ${CUDA}/ampere/kernels/kernel_decl.inc)
    set(CUDA_HEADER_FILES ${CUDA}/ampere/kernels/kernel_decl.inc)
  else()
    set(CUDA_HEADER_FILES)
  endif()

  set(CUDA_HEADER_FILES
      ${CUDA_HEADER_FILES}
      ${CUDA}/ampere/ampere_gemm_kernel.h)

  set(CUDA_SOURCES
      ${CUDA}/cuda_backend.cu
      ${CUDA_KERNELS})
else()
  set(CUDA_HEADER_FILES)
endif()

set(HEADER_FILES
    ${SRC}/opt_gemm.h
    )

set(SOURCES
    ${CUDA_SOURCES}
    ${SRC}/opt_gemm.cpp)

if (PYMODULE)
  ## PyBind ##
  add_subdirectory(pybind11)
  pybind11_add_module(OptGemm
    SHARED
      ${SOURCES}
      ${SRC}/../pyoptgemm/pyoptgemm.cpp
      ${HEADER_FILES}
      ${KERNEL_HEADER_FILES}
      ${CUDA_HEADER_FILES})
  #install(TARGETS OptGemm DESTINATION ${SKBUILD_PROJECT_NAME})
else()
  add_library(OptGemm
    SHARED
      ${SOURCES}
      ${HEADER_FILES}
      ${KERNEL_HEADER_FILES}
      ${CUDA_HEADER_FILES}
  )
  add_subdirectory(tests/)
  
  set_target_properties(OptGemm PROPERTIES PUBLIC_HEADER PUBLIC_HEADERS)
  install(TARGETS OptGemm)
endif()

target_include_directories(OptGemm
                           PRIVATE
                              ${SRC}/
                              ${GEMM_LIB_DIR}/
                           PUBLIC
                              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)