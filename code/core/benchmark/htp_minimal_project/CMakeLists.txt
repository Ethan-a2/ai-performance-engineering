cmake_minimum_required(VERSION 3.14.3)

project(aisp_htp_minimal C CXX ASM)
enable_language(ASM)
cmake_policy(SET CMP0115 OLD)

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()

set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${HEXAGON_SDK_ROOT}/utils/examples
)

if(${OS_TYPE} MATCHES "HLOS")
    add_library(aisp_htp_minimal SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/aisp_htp_minimal_stub
    )

    add_executable(aisp_htp_minimal_device
        ${CMAKE_CURRENT_SOURCE_DIR}/src/aisp_htp_minimal_host.cpp
    )

    set_target_properties(aisp_htp_minimal_device PROPERTIES OUTPUT_NAME "aisp_htp_minimal")

    target_include_directories(aisp_htp_minimal PUBLIC ${common_incs})
    target_include_directories(aisp_htp_minimal_device PUBLIC ${common_incs})

    build_idl(inc/aisp_htp_minimal.idl aisp_htp_minimal)

    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_custom_library(aisp_htp_minimal ${dsprpc})
    link_custom_library(aisp_htp_minimal_device ${dsprpc})

    set_common_compile_and_link_options(aisp_htp_minimal_device)
    target_link_libraries(aisp_htp_minimal_device aisp_htp_minimal)

    copy_binaries(aisp_htp_minimal_device aisp_htp_minimal)
else()
    set(skel_srcs
        ${CMAKE_CURRENT_BINARY_DIR}/aisp_htp_minimal_skel
        ${CMAKE_CURRENT_SOURCE_DIR}/src/aisp_htp_minimal_imp.c
    )

    add_library(aisp_htp_minimal_skel_OBJS OBJECT ${skel_srcs})
    add_library(aisp_htp_minimal_skel SHARED $<TARGET_OBJECTS:aisp_htp_minimal_skel_OBJS>)
    set_target_properties(aisp_htp_minimal_skel PROPERTIES OUTPUT_NAME "aisp_htp_minimal-${DSP_VERSION}")

    target_include_directories(aisp_htp_minimal_skel_OBJS PUBLIC ${common_incs})
    target_include_directories(aisp_htp_minimal_skel PUBLIC ${common_incs})

    build_idl(inc/aisp_htp_minimal.idl aisp_htp_minimal_skel_OBJS)
    copy_binaries(aisp_htp_minimal_skel)
endif()
