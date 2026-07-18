# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

# Locate XRT (AMD/Xilinx Runtime). Order of preference:
#   1. XRT's own CMake package (ships with the XRT dev install / Ryzen AI SDK)
#   2. Manual discovery via XILINX_XRT / common install prefixes
#
# On Windows, set -DXILINX_XRT=<path to XRT install> (or have %XILINX_XRT% set).
# On Linux, /opt/xilinx/xrt or /usr/local are searched.

set(GGML_XRT_SEARCH_PATHS
    $ENV{XILINX_XRT}
    ${XILINX_XRT}
    /opt/xilinx/xrt
    /usr/local
    )

find_package(XRT QUIET)

if (XRT_FOUND)
    message(STATUS "ggml-xrt: found XRT via CMake package (${XRT_INCLUDE_DIRS})")
    set(GGML_XRT_INCLUDE_DIRS ${XRT_INCLUDE_DIRS})
    set(GGML_XRT_LIBRARIES    XRT::xrt_coreutil)
else()
    find_path(GGML_XRT_INCLUDE_DIRS
        NAMES xrt/xrt_device.h
        HINTS ${GGML_XRT_SEARCH_PATHS}
        PATH_SUFFIXES include)

    find_library(GGML_XRT_COREUTIL
        NAMES xrt_coreutil
        HINTS ${GGML_XRT_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64)

    if (NOT GGML_XRT_INCLUDE_DIRS OR NOT GGML_XRT_COREUTIL)
        message(FATAL_ERROR
            "ggml-xrt: could not locate XRT. Set -DXILINX_XRT=<path> or install the XRT dev package. "
            "include=${GGML_XRT_INCLUDE_DIRS} coreutil=${GGML_XRT_COREUTIL}")
    endif()

    message(STATUS "ggml-xrt: found XRT headers at ${GGML_XRT_INCLUDE_DIRS}, coreutil ${GGML_XRT_COREUTIL}")
    set(GGML_XRT_LIBRARIES ${GGML_XRT_COREUTIL})
endif()

ggml_add_backend_library(ggml-xrt
                         ggml-xrt.cpp
                         ../../include/ggml-xrt.h
                         )

target_include_directories(ggml-xrt PRIVATE . .. ${GGML_XRT_INCLUDE_DIRS})
target_link_libraries(ggml-xrt PRIVATE ggml-base ${GGML_XRT_LIBRARIES})

if (MSVC)
    # XRT public headers pick std::any (C++17) over boost::any only when
    # __cplusplus reports >= 201703L; MSVC leaves it at 199711L without this.
    target_compile_options(ggml-xrt PRIVATE /Zc:__cplusplus)
endif()

add_compile_definitions(GGML_USE_XRT)
