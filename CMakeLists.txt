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

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(SLEIPNIR_BUILD_BENCHMARKS "Build CasADi and Sleipnir benchmarks" OFF)
option(SLEIPNIR_BUILD_EXAMPLES "Build examples" OFF)
option(SLEIPNIR_BUILD_PYTHON "Build Python module" OFF)
option(
    SLEIPNIR_DISABLE_DIAGNOSTICS
    "Disable diagnostics support at compile-time"
    OFF
)
option(
    SLEIPNIR_ENABLE_BOUND_PROJECTION
    "Enable projecting the state onto bound inequality constraints inside problem setup"
    OFF
)

# Options for using a package manager (e.g., vcpkg) for certain dependencies
option(SLEIPNIR_USE_SYSTEM_BENCHMARK "Use system benchmark" OFF)
option(SLEIPNIR_USE_SYSTEM_EIGEN "Use system eigen" OFF)
option(SLEIPNIR_USE_SYSTEM_NANOBIND "Use system nanobind" OFF)

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

if(SLEIPNIR_DISABLE_DIAGNOSTICS)
    target_compile_definitions(Sleipnir PUBLIC SLEIPNIR_DISABLE_DIAGNOSTICS)
endif()

if(SLEIPNIR_ENABLE_BOUND_PROJECTION)
    target_compile_definitions(Sleipnir PUBLIC SLEIPNIR_ENABLE_BOUND_PROJECTION)
endif()

# Eigen dependency
if(NOT SLEIPNIR_USE_SYSTEM_EIGEN)
    set(EIGEN_BUILD_CMAKE_PACKAGE TRUE)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.1
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
    # main on 2026-08-29
    GIT_TAG efcab60e5360b699f45d2a44dcb9a79bba40c356
    SYSTEM
)
FetchContent_MakeAvailable(small_vector)

target_link_libraries(Sleipnir PUBLIC small_vector)

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
    # Catch2 dependency
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.16.0
        GIT_SHALLOW ON
        EXCLUDE_FROM_ALL
        SYSTEM
    )
    FetchContent_MakeAvailable(Catch2)

    # Enable StringMaker<std::string_view>
    target_compile_features(Catch2 PRIVATE cxx_std_23)
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
install(DIRECTORY include/ COMPONENT Sleipnir DESTINATION include)
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
    INSTALL_DESTINATION lib/cmake/Sleipnir
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
if(SLEIPNIR_BUILD_BENCHMARKS)
    # Autodiff and optimization benchmarks (if CasADi exists)
    find_package(casadi QUIET)
    if(casadi_FOUND)
        # benchmark dependency
        option(
            BENCHMARK_ENABLE_TESTING
            "Enable testing of the benchmark library."
            OFF
        )
        option(
            BENCHMARK_ENABLE_WERROR
            "Build Release candidates with -Werror."
            OFF
        )
        option(BENCHMARK_ENABLE_INSTALL "Enable installation of benchmark." OFF)
        if(NOT SLEIPNIR_USE_SYSTEM_BENCHMARK)
            FetchContent_Declare(
                benchmark
                GIT_REPOSITORY https://github.com/google/benchmark.git
                GIT_TAG v1.9.5
                GIT_SHALLOW ON
                EXCLUDE_FROM_ALL
                SYSTEM
            )
            FetchContent_MakeAvailable(benchmark)
        else()
            find_package(benchmark CONFIG REQUIRED)
        endif()

        # Autodiff benchmarks
        foreach(benchmark "gradient" "jacobian" "hessian")
            file(
                GLOB ${benchmark}_autodiff_benchmark_src
                benchmarks/${benchmark}_autodiff/*.cpp
            )
            add_executable(
                ${benchmark}_autodiff_benchmark
                ${${benchmark}_autodiff_benchmark_src}
            )
            compiler_flags(${benchmark}_autodiff_benchmark)
            target_include_directories(
                ${benchmark}_autodiff_benchmark
                PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
            )
            target_link_libraries(
                ${benchmark}_autodiff_benchmark
                PUBLIC Sleipnir casadi benchmark::benchmark
            )
        endforeach()

        # Optimization benchmarks
        foreach(benchmark "cart_pole" "flywheel")
            file(
                GLOB ${benchmark}_optimization_benchmark_src
                benchmarks/${benchmark}_optimization/*.cpp
            )
            add_executable(
                ${benchmark}_optimization_benchmark
                ${${benchmark}_optimization_benchmark_src}
            )
            compiler_flags(${benchmark}_optimization_benchmark)
            target_include_directories(
                ${benchmark}_optimization_benchmark
                PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
            )
            target_link_libraries(
                ${benchmark}_optimization_benchmark
                PUBLIC Sleipnir casadi benchmark::benchmark
            )
        endforeach()
    endif()
endif()

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
    enable_testing()
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    include(Catch)

    # Build autodiff explicit instantiations shared by autodiff and
    # optimization tests
    file(
        GLOB_RECURSE sleipnir_autodiff_explicit_double_src
        test/src/autodiff/*_explicit_double.cpp
    )
    add_library(
        sleipnir_autodiff_explicit_double
        OBJECT
        ${sleipnir_autodiff_explicit_double_src}
    )
    compiler_flags(sleipnir_autodiff_explicit_double)
    target_include_directories(
        sleipnir_autodiff_explicit_double
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    )
    target_link_libraries(sleipnir_autodiff_explicit_double PUBLIC Sleipnir)

    # Build Sleipnir autodiff tests
    file(GLOB_RECURSE sleipnir_autodiff_test_src test/src/autodiff/*_test.cpp)
    add_executable(sleipnir_autodiff_test ${sleipnir_autodiff_test_src})
    compiler_flags(sleipnir_autodiff_test)
    target_include_directories(
        sleipnir_autodiff_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    )
    target_link_libraries(
        sleipnir_autodiff_test
        PUBLIC Sleipnir sleipnir_autodiff_explicit_double Catch2::Catch2WithMain
    )

    # Build Sleipnir optimization tests
    file(
        GLOB_RECURSE sleipnir_optimization_test_src
        test/src/optimization/*_explicit_double.cpp
        test/src/optimization/*_test.cpp
    )
    add_executable(sleipnir_optimization_test ${sleipnir_optimization_test_src})
    compiler_flags(sleipnir_optimization_test)
    target_include_directories(
        sleipnir_optimization_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    )
    target_link_libraries(
        sleipnir_optimization_test
        PUBLIC Sleipnir sleipnir_autodiff_explicit_double Catch2::Catch2WithMain
    )

    # Build Sleipnir util tests
    file(GLOB_RECURSE sleipnir_util_test_src test/src/util/*_test.cpp)
    add_executable(sleipnir_util_test ${sleipnir_util_test_src})
    compiler_flags(sleipnir_util_test)
    target_include_directories(
        sleipnir_util_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    )
    target_link_libraries(
        sleipnir_util_test
        PUBLIC Sleipnir Catch2::Catch2WithMain
    )

    if(NOT CMAKE_TOOLCHAIN_FILE)
        catch_discover_tests(sleipnir_autodiff_test)
        catch_discover_tests(sleipnir_optimization_test)
        catch_discover_tests(sleipnir_util_test)
    endif()
endif()

# Build examples and example tests
if(SLEIPNIR_BUILD_EXAMPLES)
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
        target_link_libraries(${example} PUBLIC Sleipnir)

        # Build example test if files exist for it
        if(
            PROJECT_IS_TOP_LEVEL
            AND BUILD_TESTING
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
                PUBLIC Sleipnir Catch2::Catch2WithMain
            )
            if(NOT CMAKE_TOOLCHAIN_FILE)
                catch_discover_tests(${example}_test)
            endif()
        endif()
    endforeach()
endif()

if(SLEIPNIR_BUILD_PYTHON)
    find_package(
        Python
        REQUIRED
        COMPONENTS Interpreter Development.Module Development.SABIModule
    )
    if(DEFINED PY_BUILD_CMAKE_MODULE_NAME)
        set(PY_DEST ${PY_BUILD_CMAKE_MODULE_NAME})
    else()
        set(PY_DEST
            lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/sleipnir
        )
    endif()

    # nanobind dependency
    if(NOT SLEIPNIR_USE_SYSTEM_NANOBIND)
        FetchContent_Declare(
            nanobind
            GIT_REPOSITORY https://github.com/wjakob/nanobind.git
            GIT_TAG v3.1.0
            GIT_SHALLOW ON
            EXCLUDE_FROM_ALL
            SYSTEM
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

    install(
        TARGETS _sleipnir
        COMPONENT python_modules
        LIBRARY DESTINATION ${PY_DEST}
    )

    # py-build-cmake copies the pure Python sources into the wheel itself.
    # Install them for plain CMake installs as well.
    if(NOT DEFINED PY_BUILD_CMAKE_MODULE_NAME)
        install(
            DIRECTORY python/src/sleipnir/
            DESTINATION ${PY_DEST}
            COMPONENT python_modules
            FILES_MATCHING
            PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE
        )
    endif()

    # Generate stubs at install time because only the install tree has the
    # sleipnir/ package layout needed to import the module by its fully
    # qualified name `sleipnir._sleipnir`.
    #
    # Importing sleipnir._sleipnir requires the directory containing the
    # sleipnir package (i.e., PY_DEST's parent) to be on the Python path. For
    # py-build-cmake builds, PY_DEST has no parent. For plain CMake installs,
    # PY_DEST's parent is lib/pythonX.Y.
    cmake_path(GET PY_DEST PARENT_PATH PY_DEST_PARENT)
    nanobind_add_stub(
        _sleipnir_stub
        INSTALL_TIME
        MARKER_FILE ${PY_DEST}/py.typed
        MODULE sleipnir._sleipnir
        OUTPUT ${PY_DEST}/_sleipnir/__init__.pyi
        PYTHON_PATH ${PY_DEST_PARENT}
        DEPENDS _sleipnir
        COMPONENT python_modules
    )
    nanobind_add_stub(
        _sleipnir_autodiff_stub
        INSTALL_TIME
        MODULE sleipnir._sleipnir.autodiff
        OUTPUT ${PY_DEST}/_sleipnir/autodiff.pyi
        PYTHON_PATH ${PY_DEST_PARENT}
        DEPENDS _sleipnir
        COMPONENT python_modules
    )
    nanobind_add_stub(
        _sleipnir_optimization_stub
        INSTALL_TIME
        MODULE sleipnir._sleipnir.optimization
        OUTPUT ${PY_DEST}/_sleipnir/optimization.pyi
        PYTHON_PATH ${PY_DEST_PARENT}
        DEPENDS _sleipnir
        COMPONENT python_modules
    )
    install(
        CODE
            "execute_process(
                COMMAND \"${Python_EXECUTABLE}\"
                    \"${CMAKE_CURRENT_SOURCE_DIR}/python/postprocess_stubs.py\"
                    ${PY_DEST}/_sleipnir/autodiff.pyi
                    ${PY_DEST}/_sleipnir/optimization.pyi
                WORKING_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}\"
                COMMAND_ERROR_IS_FATAL ANY
            )"
        COMPONENT python_modules
    )

    if(NOT CMAKE_CROSSCOMPILING)
        set(sleipnir_headers
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/autodiff/expression_type.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/autodiff/gradient.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/autodiff/hessian.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/autodiff/jacobian.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/autodiff/variable.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/autodiff/variable_block.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/autodiff/variable_matrix.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/optimization/ocp.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/optimization/ocp/dynamics_type.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/optimization/ocp/timestep_method.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/optimization/ocp/transcription_method.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/optimization/problem.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/optimization/solver/exit_status.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/optimization/solver/iteration_info.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/sleipnir/optimization/solver/options.hpp
        )

        # Generate docs for the Python module
        include(Pybind11Mkdoc)
        pybind11_mkdoc(_sleipnir "${sleipnir_headers}")
        add_dependencies(_sleipnir _sleipnir_docstrings)
    endif()
endif()
