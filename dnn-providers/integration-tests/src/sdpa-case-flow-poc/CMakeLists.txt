# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT HIPDNN_ENABLE_SDPA)
    message(FATAL_ERROR
        "HIPDNN_IT_ENABLE_SDPA_CASE_FLOW_POC needs hipDNN built with HIPDNN_ENABLE_SDPA=ON")
endif()

set(SDPA_CASE_FLOW_DISCOVERY_EXE hipdnn_sdpa_case_flow_discovery)

add_executable(${SDPA_CASE_FLOW_DISCOVERY_EXE}
    Main.cpp
    CaseGraph.cpp
)

target_link_libraries(${SDPA_CASE_FLOW_DISCOVERY_EXE}
    PRIVATE
        hip::host
        hipdnn_test_sdk
        Threads::Threads
)

# Warnings are reported but do not fail the build.
target_compile_options(${SDPA_CASE_FLOW_DISCOVERY_EXE}
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS} -Wno-error)

set_target_properties(${SDPA_CASE_FLOW_DISCOVERY_EXE}
    PROPERTIES
        BUILD_RPATH_USE_ORIGIN TRUE
)
