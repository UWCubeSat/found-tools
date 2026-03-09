cmake_minimum_required(VERSION 3.21)

# ---- Project variables ----
set(FOUND_TOOLS_PROJECT_NAME  found_tools)
set(FOUND_REPO_URL            https://github.com/UWCubeSat/found.git)
set(FOUND_REPO_TAG            main)
set(FOUND_FETCHCONTENT_NAME   found_external)
set(FOUND_LIB_TARGET          found::found_lib)
set(PYBIND11_PACKAGE_NAME     pybind11)
set(BINDING_MODULE_NAME       _cameraGeometry)
set(BINDING_SRC               src/found_CLI_tools/cameraGeometry/binding.cpp)
set(INSTALL_DEST              found_CLI_tools/cameraGeometry)

project(${FOUND_TOOLS_PROJECT_NAME} LANGUAGES CXX)

# 1. Pull in the external C++ project
include(FetchContent)
FetchContent_Declare(
  ${FOUND_FETCHCONTENT_NAME}
  GIT_REPOSITORY ${FOUND_REPO_URL}
  GIT_TAG        ${FOUND_REPO_TAG}
)
FetchContent_MakeAvailable(${FOUND_FETCHCONTENT_NAME})

# 2. Pull in pybind11
find_package(${PYBIND11_PACKAGE_NAME} REQUIRED)

# 3. Create the wrapper module
pybind11_add_module(${BINDING_MODULE_NAME} ${BINDING_SRC})

# 4. Link the external library
target_link_libraries(${BINDING_MODULE_NAME} PRIVATE ${FOUND_LIB_TARGET})

install(TARGETS ${BINDING_MODULE_NAME} DESTINATION ${INSTALL_DEST})