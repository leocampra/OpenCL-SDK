add_executable(compute_tests main.cpp)
target_link_libraries(compute_tests PRIVATE OpenCL::Compute)
target_compile_definitions(compute_tests
  PRIVATE
    CL_HPP_TARGET_OPENCL_VERSION=300
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    CL_TARGET_OPENCL_VERSION=300
)
set_target_properties(compute_tests
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    FOLDER "Tests/Compute"
)
add_test(
  NAME compute_tests
  COMMAND compute_tests
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# The C interface, tested from C so the header is checked as a C header.
add_executable(compute_capi_tests capi.c)
target_link_libraries(compute_capi_tests PRIVATE OpenCL::Compute)
target_compile_definitions(compute_capi_tests
  PRIVATE
    CL_TARGET_OPENCL_VERSION=300
)
set_target_properties(compute_capi_tests
  PROPERTIES
    C_STANDARD 99
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    FOLDER "Tests/Compute"
)
add_test(
  NAME compute_capi_tests
  COMMAND compute_capi_tests
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
