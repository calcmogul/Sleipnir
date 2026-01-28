cmake_minimum_required(VERSION 3.21)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build macOS universal libraries by default
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "")
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "")

project(Sleipnir LANGUAGES CXX)

include(BuildTypes)
include(CMakePackageConfigHelpers)
include(CTest)
include(CompilerFlags)
include(FetchContent)

# Control where the static and shared libraries are built so that on Windows,
# we don't need to tinker with the path to run the executable
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

option(BUILD_BENCHMARKS "Build CasADi and Sleipnir benchmarks" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_PYTHON "Build Python module" OFF)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(DISABLE_DIAGNOSTICS "Disable diagnostics support at compile-time" OFF)
option(
    ENABLE_BOUND_PROJECTION
    "Enable projecting the state onto bound inequality constraints inside problem setup"
    OFF
)

# Options for using a package manager (e.g., vcpkg) for certain dependencies
option(USE_SYSTEM_EIGEN "Use system eigen" OFF)
option(USE_SYSTEM_NANOBIND "Use system nanobind" OFF)

file(GLOB_RECURSE Sleipnir_src src/*.cpp)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS FALSE)
add_library(Sleipnir ${Sleipnir_src})
add_library(Sleipnir::Sleipnir ALIAS Sleipnir)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

compiler_flags(Sleipnir)
target_compile_definitions(Sleipnir PRIVATE SLEIPNIR_EXPORTS)
target_include_directories(
    Sleipnir
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Required for std::async()
find_package(Threads)
if(Threads_FOUND)
    target_link_libraries(Sleipnir PUBLIC Threads::Threads)
endif()

if(DISABLE_DIAGNOSTICS)
    target_compile_definitions(Sleipnir PUBLIC SLEIPNIR_DISABLE_DIAGNOSTICS)
endif()

if(ENABLE_BOUND_PROJECTION)
    target_compile_definitions(Sleipnir PUBLIC SLEIPNIR_ENABLE_BOUND_PROJECTION)
endif()

# Eigen dependency
if(NOT USE_SYSTEM_EIGEN)
    set(EIGEN_BUILD_CMAKE_PACKAGE TRUE)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        # master on 2026-01-14
        GIT_TAG 251bff28859af2ed2d5bdf14034175f03cafffc7
        SYSTEM
    )
    FetchContent_MakeAvailable(Eigen3)
else()
    find_package(Eigen3 CONFIG REQUIRED)
endif()

target_link_libraries(Sleipnir PUBLIC Eigen3::Eigen)

# small_vector dependency
FetchContent_Declare(
    small_vector
    GIT_REPOSITORY https://github.com/gharveymn/small_vector.git
    # main on 2025-05-15
    GIT_TAG b19a9c477cad5e58f8aed59dbe9be1c30e4b9826
    SYSTEM
)
FetchContent_MakeAvailable(small_vector)

target_link_libraries(Sleipnir PUBLIC small_vector)

if(BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
    # Catch2 dependency
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.11.0
        GIT_SHALLOW ON
        EXCLUDE_FROM_ALL
        SYSTEM
    )
    FetchContent_MakeAvailable(Catch2)
endif()

install(
    TARGETS Sleipnir
    COMPONENT Sleipnir
    EXPORT SleipnirTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
export(TARGETS Sleipnir FILE Sleipnir.cmake NAMESPACE Sleipnir::)
install(DIRECTORY include/ COMPONENT Sleipnir DESTINATION "include")
install(
    EXPORT SleipnirTargets
    FILE Sleipnir.cmake
    NAMESPACE Sleipnir::
    DESTINATION lib/cmake/Sleipnir
)

# Generate the config file that includes the exports
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/SleipnirConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SleipnirConfig.cmake
    INSTALL_DESTINATION "lib/cmake/Sleipnir"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Install the config file
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/SleipnirConfig.cmake
    COMPONENT Sleipnir
    DESTINATION lib/cmake/Sleipnir
)

# Add benchmark executables
if(BUILD_BENCHMARKS)
    # Perf benchmark
    foreach(benchmark "cart_pole" "flywheel")
        file(
            GLOB ${benchmark}_perf_benchmark_src
            benchmarks/perf/*.cpp
            benchmarks/perf/${benchmark}/*.cpp
        )
        add_executable(
            ${benchmark}_perf_benchmark
            ${${benchmark}_perf_benchmark_src}
        )
        compiler_flags(${benchmark}_perf_benchmark)
        target_include_directories(
            ${benchmark}_perf_benchmark
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
                ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/perf
        )
        target_link_libraries(${benchmark}_perf_benchmark Sleipnir)
    endforeach()

    # Scalability benchmark (if CasADi exists)
    find_package(casadi QUIET)
    if(casadi_FOUND)
        foreach(benchmark "cart_pole" "flywheel")
            file(
                GLOB ${benchmark}_scalability_benchmark_src
                benchmarks/scalability/*.cpp
                benchmarks/scalability/${benchmark}/*.cpp
            )
            add_executable(
                ${benchmark}_scalability_benchmark
                ${${benchmark}_scalability_benchmark_src}
            )
            compiler_flags(${benchmark}_scalability_benchmark)
            target_include_directories(
                ${benchmark}_scalability_benchmark
                PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
                    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/scalability
            )
            target_link_libraries(
                ${benchmark}_scalability_benchmark
                Sleipnir
                casadi
            )
        endforeach()
    endif()
endif()

if(BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
    enable_testing()
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    include(Catch)

    # Build Sleipnir tests
    file(GLOB_RECURSE sleipnir_test_src test/src/*.cpp)
    add_executable(sleipnir_test ${sleipnir_test_src})
    compiler_flags(sleipnir_test)
    target_include_directories(
        sleipnir_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    )
    target_link_libraries(sleipnir_test Sleipnir Catch2::Catch2WithMain)
    if(NOT CMAKE_TOOLCHAIN_FILE)
        catch_discover_tests(sleipnir_test)
    endif()
endif()

# Build examples and example tests
if(BUILD_EXAMPLES)
    include(SubdirList)
    subdir_list(EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    foreach(example ${EXAMPLES})
        # Build example
        file(GLOB_RECURSE sources examples/${example}/src/*.cpp)
        add_executable(${example} ${sources})
        compiler_flags(${example})
        target_include_directories(
            ${example}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/include
        )
        target_link_libraries(${example} Sleipnir)

        # Build example test if files exist for it
        if(
            BUILD_TESTING
            AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/test
        )
            file(GLOB_RECURSE test_sources examples/${example}/test/*.cpp)
            add_executable(${example}_test ${sources} ${test_sources})
            compiler_flags(${example}_test)
            target_compile_definitions(${example}_test PUBLIC RUNNING_TESTS)
            target_include_directories(
                ${example}_test
                PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/test/include
            )
            target_link_libraries(
                ${example}_test
                Sleipnir
                Catch2::Catch2WithMain
            )
            if(NOT CMAKE_TOOLCHAIN_FILE)
                catch_discover_tests(${example}_test)
            endif()
        endif()
    endforeach()
endif()

if(BUILD_PYTHON)
    find_package(
        Python
        REQUIRED
        COMPONENTS Interpreter Development.Module Development.SABIModule
    )
    if(DEFINED PY_BUILD_CMAKE_MODULE_NAME)
        set(PY_DEST ${PY_BUILD_CMAKE_MODULE_NAME})
    else()
        set(PY_DEST lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
    endif()

    # nanobind dependency
    if(NOT USE_SYSTEM_NANOBIND)
        FetchContent_Declare(
            nanobind
            GIT_REPOSITORY https://github.com/wjakob/nanobind.git
            GIT_TAG v2.11.0
            GIT_SHALLOW ON
            EXCLUDE_FROM_ALL
            SYSTEM
            PATCH_COMMAND
                git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nanobind.patch
            UPDATE_DISCONNECTED 1
        )
        FetchContent_MakeAvailable(nanobind)
    else()
        find_package(nanobind CONFIG REQUIRED)
    endif()

    file(GLOB_RECURSE python_ext_src python/cpp/*.cpp)

    # Build Sleipnir dependency directly into the wheel to avoid having to
    # configure RPATHs
    nanobind_add_module(_sleipnir STABLE_ABI ${python_ext_src} ${Sleipnir_src})
    compiler_flags(_sleipnir)
    target_compile_definitions(_sleipnir PRIVATE SLEIPNIR_PYTHON=1)
    target_include_directories(
        _sleipnir
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp
    )
    target_link_libraries(
        _sleipnir
        PUBLIC Threads::Threads Eigen3::Eigen small_vector
    )

    # Suppress compiler warnings in nanobind
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        # nanobind/include/nanobind/nb_attr.h:212:14: warning: ISO C++ forbids zero-size array
        #  212 |     arg_data args[Size];
        #      |              ^~~~
        target_compile_options(_sleipnir PRIVATE -Wno-pedantic)
    endif()
    if(
        ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"
        OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang"
    )
        target_compile_options(nanobind-static-abi3 PRIVATE -Wno-array-bounds)
        target_compile_options(
            _sleipnir
            PRIVATE -Wno-nested-anon-types -Wno-zero-length-array
        )
    endif()

    install(
        TARGETS _sleipnir
        COMPONENT python_modules
        LIBRARY DESTINATION ${PY_DEST}
    )

    nanobind_add_stub(
        _sleipnir_stub
        INSTALL_TIME
        MARKER_FILE sleipnir/py.typed
        MODULE _sleipnir
        OUTPUT sleipnir/__init__.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:_sleipnir>
        DEPENDS _sleipnir
        COMPONENT python_modules
    )
    nanobind_add_stub(
        _sleipnir_autodiff_stub
        INSTALL_TIME
        MODULE _sleipnir.autodiff
        OUTPUT sleipnir/autodiff/__init__.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:_sleipnir>
        DEPENDS _sleipnir
        COMPONENT python_modules
    )
    nanobind_add_stub(
        _sleipnir_optimization_stub
        INSTALL_TIME
        MODULE _sleipnir.optimization
        OUTPUT sleipnir/optimization/__init__.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:_sleipnir>
        DEPENDS _sleipnir
        COMPONENT python_modules
    )

    # pybind11_mkdoc doesn't support Windows
    if(NOT WIN32 AND NOT CMAKE_CROSSCOMPILING)
        # pybind11_mkdoc dependency
        FetchContent_Declare(
            pybind11_mkdoc
            GIT_REPOSITORY https://github.com/pybind/pybind11_mkdoc.git
            # master on 2023-02-08
            GIT_TAG 42fbf377824185e255b06d68fa70f4efcd569e2d
            GIT_SUBMODULES ""
            EXCLUDE_FROM_ALL
            SYSTEM
        )
        FetchContent_MakeAvailable(pybind11_mkdoc)

        file(
            GLOB_RECURSE sleipnir_headers
            include/sleipnir/autodiff/expression_type.hpp
            include/sleipnir/autodiff/gradient.hpp
            include/sleipnir/autodiff/hessian.hpp
            include/sleipnir/autodiff/jacobian.hpp
            include/sleipnir/autodiff/variable.hpp
            include/sleipnir/autodiff/variable_block.hpp
            include/sleipnir/autodiff/variable_matrix.hpp
            include/sleipnir/optimization/ocp.hpp
            include/sleipnir/optimization/ocp/dynamics_type.hpp
            include/sleipnir/optimization/ocp/timestep_method.hpp
            include/sleipnir/optimization/ocp/transcription_method.hpp
            include/sleipnir/optimization/problem.hpp
            include/sleipnir/optimization/solver/exit_status.hpp
            include/sleipnir/optimization/solver/iteration_info.hpp
            include/sleipnir/optimization/solver/options.hpp
        )

        # Generate docs for the Python module
        include(Pybind11Mkdoc)
        pybind11_mkdoc(_sleipnir "${sleipnir_headers}")
        add_dependencies(_sleipnir _sleipnir_docstrings)
    endif()
endif()

if(EMSCRIPTEN)
    file(GLOB_RECURSE js_ext_src js/cpp/*.cpp)

    # Build Sleipnir dependency directly into the wheel to avoid having to
    # configure RPATHs
    add_library(_sleipnir ${js_ext_src} ${Sleipnir_src})
    compiler_flags(_sleipnir)
    target_include_directories(
        _sleipnir
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/js/cpp
    )
    target_link_libraries(
        _sleipnir
        PUBLIC Threads::Threads Eigen3::Eigen small_vector
    )
endif()
