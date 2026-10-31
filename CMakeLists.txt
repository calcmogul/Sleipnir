cmake_minimum_required(VERSION 3.28)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build macOS universal libraries by default
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "")
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "")

set(SLEIPNIR_VERSION 0.0.0)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    execute_process(
        COMMAND git describe --match "v[0-9]*"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE git_describe
        RESULT_VARIABLE git_describe_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(
        git_describe_result EQUAL 0
        AND git_describe MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+)"
    )
        set(SLEIPNIR_VERSION ${CMAKE_MATCH_1})
    endif()
endif()

project(Sleipnir VERSION ${SLEIPNIR_VERSION} LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(CTest)
include(CompilerFlags)
include(FetchContent)

if(PROJECT_IS_TOP_LEVEL)
    include(BuildTypes)

    # Control where the static and shared libraries are built so that on
    # Windows, we don't need to tinker with the path to run the executable
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()

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
set(SLEIPNIR_TEST_SCALAR_TYPES
    "double"
    CACHE STRING
    "Semicolon-separated list of scalar types to test (double, ExplicitDouble)"
)

# Options for using a package manager (e.g., vcpkg) for certain dependencies
option(SLEIPNIR_USE_SYSTEM_BENCHMARK "Use system benchmark" OFF)
option(SLEIPNIR_USE_SYSTEM_EIGEN "Use system eigen" OFF)
option(SLEIPNIR_USE_SYSTEM_NANOBIND "Use system nanobind" OFF)

file(GLOB_RECURSE Sleipnir_src CONFIGURE_DEPENDS src/*.cpp)

add_library(Sleipnir ${Sleipnir_src})
add_library(Sleipnir::Sleipnir ALIAS Sleipnir)
set_target_properties(Sleipnir PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)

compiler_flags(Sleipnir)
target_compile_definitions(Sleipnir PRIVATE SLEIPNIR_EXPORTS)
file(GLOB_RECURSE Sleipnir_headers CONFIGURE_DEPENDS include/*.hpp)
target_sources(
    Sleipnir
    PUBLIC FILE_SET HEADERS BASE_DIRS include FILES ${Sleipnir_headers}
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
        # master on 2026-10-17
        GIT_TAG 3c3bf4532bf9e059e3999827b00b7452c2bcde7d
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

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    # Catch2 dependency
    if(CMAKE_SYSTEM_NAME STREQUAL "Generic")
        # Bare-metal targets don't support POSIX signals
        set(CATCH_CONFIG_NO_POSIX_SIGNALS ON)
    endif()
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
    FILE_SET HEADERS DESTINATION include
)
export(TARGETS Sleipnir FILE Sleipnir.cmake NAMESPACE Sleipnir::)
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

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SleipnirConfigVersion.cmake
    COMPATIBILITY SameMinorVersion
)

# Install the config files
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SleipnirConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/SleipnirConfigVersion.cmake
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
                CONFIGURE_DEPENDS
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
        foreach(benchmark "cart_pole" "double_integrator" "flywheel")
            file(
                GLOB ${benchmark}_optimization_benchmark_src
                CONFIGURE_DEPENDS
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

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    enable_testing()
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    include(Catch)

    set(sleipnir_allowed_test_scalar_types double ExplicitDouble)
    foreach(type IN LISTS SLEIPNIR_TEST_SCALAR_TYPES)
        if(NOT type IN_LIST sleipnir_allowed_test_scalar_types)
            message(
                FATAL_ERROR
                "Unknown scalar type \"${type}\" in SLEIPNIR_TEST_SCALAR_TYPES"
            )
        endif()
    endforeach()
    if(NOT SLEIPNIR_TEST_SCALAR_TYPES)
        message(FATAL_ERROR "SLEIPNIR_TEST_SCALAR_TYPES is empty")
    endif()
    list(JOIN SLEIPNIR_TEST_SCALAR_TYPES "," sleipnir_test_scalar_types)

    if("ExplicitDouble" IN_LIST SLEIPNIR_TEST_SCALAR_TYPES)
        # Build autodiff explicit instantiations
        file(
            GLOB_RECURSE sleipnir_autodiff_explicit_src
            CONFIGURE_DEPENDS
            test/src/autodiff/*_explicit_double.cpp
        )
        add_library(
            sleipnir_autodiff_explicit
            OBJECT
            ${sleipnir_autodiff_explicit_src}
        )
        compiler_flags(sleipnir_autodiff_explicit)
        target_include_directories(
            sleipnir_autodiff_explicit
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/include
        )
        target_link_libraries(sleipnir_autodiff_explicit PUBLIC Sleipnir)

        # Build optimization explicit instantiations
        file(
            GLOB_RECURSE sleipnir_optimization_explicit_src
            CONFIGURE_DEPENDS
            test/src/optimization/*_explicit_double.cpp
        )
        add_library(
            sleipnir_optimization_explicit
            OBJECT
            ${sleipnir_optimization_explicit_src}
        )
        compiler_flags(sleipnir_optimization_explicit)
        target_include_directories(
            sleipnir_optimization_explicit
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/include
        )
        target_link_libraries(sleipnir_optimization_explicit PUBLIC Sleipnir)
    endif()

    # Build Sleipnir autodiff tests
    file(
        GLOB_RECURSE sleipnir_autodiff_test_src
        CONFIGURE_DEPENDS
        test/src/autodiff/*_test.cpp
    )
    add_executable(sleipnir_autodiff_test ${sleipnir_autodiff_test_src})
    compiler_flags(sleipnir_autodiff_test)
    target_compile_definitions(
        sleipnir_autodiff_test
        PRIVATE SCALAR_TYPES=${sleipnir_test_scalar_types}
    )
    target_include_directories(
        sleipnir_autodiff_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    )
    target_link_libraries(
        sleipnir_autodiff_test
        PUBLIC
            Sleipnir
            $<TARGET_NAME_IF_EXISTS:sleipnir_autodiff_explicit>
            Catch2::Catch2WithMain
    )

    # Build Sleipnir optimization tests
    file(
        GLOB_RECURSE sleipnir_optimization_test_src
        CONFIGURE_DEPENDS
        test/src/optimization/*_test.cpp
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Generic")
        # Bare-metal targets don't support threads
        list(
            FILTER sleipnir_optimization_test_src
            EXCLUDE
            REGEX "/multistart_test\\.cpp$"
        )
    endif()
    add_executable(sleipnir_optimization_test ${sleipnir_optimization_test_src})
    compiler_flags(sleipnir_optimization_test)
    target_compile_definitions(
        sleipnir_optimization_test
        PRIVATE SCALAR_TYPES=${sleipnir_test_scalar_types}
    )
    target_include_directories(
        sleipnir_optimization_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    )
    target_link_libraries(
        sleipnir_optimization_test
        PUBLIC
            Sleipnir
            $<TARGET_NAME_IF_EXISTS:sleipnir_autodiff_explicit>
            $<TARGET_NAME_IF_EXISTS:sleipnir_optimization_explicit>
            Catch2::Catch2WithMain
    )

    # Build Sleipnir util tests
    file(
        GLOB_RECURSE sleipnir_util_test_src
        CONFIGURE_DEPENDS
        test/src/util/*_test.cpp
    )
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

    if(NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR)
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
        file(
            GLOB_RECURSE sources
            CONFIGURE_DEPENDS
            examples/${example}/src/*.cpp
        )
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
            file(
                GLOB_RECURSE test_sources
                CONFIGURE_DEPENDS
                examples/${example}/test/*.cpp
            )
            add_executable(${example}_test ${sources} ${test_sources})
            compiler_flags(${example}_test)
            target_compile_definitions(${example}_test PRIVATE RUNNING_TESTS)
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
            if(NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR)
                catch_discover_tests(${example}_test)
            endif()
        endif()
    endforeach()
endif()

if(SLEIPNIR_BUILD_PYTHON)
    if(CMAKE_SYSTEM_NAME STREQUAL "Generic")
        message(
            FATAL_ERROR
            "SLEIPNIR_BUILD_PYTHON isn't supported on bare-metal targets"
        )
    endif()
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

    file(GLOB_RECURSE python_ext_src CONFIGURE_DEPENDS python/cpp/*.cpp)

    # Build Sleipnir directly into the module and manually inherit its compile
    # definitions, include directories, and link libraries. This avoids having
    # to configure RPATHs to a Sleipnir shared library.
    #
    # SLEIPNIR_IMPORTS is excluded because consumers may add it to Sleipnir's
    # interface compile definitions. It declares Sleipnir's symbols are defined
    # in a separate DLL, but they're compiled into the module here.
    nanobind_add_module(_sleipnir STABLE_ABI ${python_ext_src} ${Sleipnir_src})
    compiler_flags(_sleipnir)
    target_compile_definitions(
        _sleipnir
        PRIVATE
            $<FILTER:$<TARGET_PROPERTY:Sleipnir,INTERFACE_COMPILE_DEFINITIONS>,EXCLUDE,^SLEIPNIR_IMPORTS$|^SLEIPNIR_IMPORTS=>
            SLEIPNIR_PYTHON=1
    )
    target_include_directories(
        _sleipnir
        PRIVATE
            $<TARGET_PROPERTY:Sleipnir,INTERFACE_INCLUDE_DIRECTORIES>
            ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp
    )
    target_link_libraries(
        _sleipnir
        PRIVATE $<TARGET_PROPERTY:Sleipnir,INTERFACE_LINK_LIBRARIES>
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

    if(NOT CMAKE_CROSSCOMPILING)
        # Generate stubs at install time because only the install tree has the
        # sleipnir/ package layout needed to import the module by its fully
        # qualified name `sleipnir._sleipnir`.
        #
        # Importing sleipnir._sleipnir requires the directory containing the
        # sleipnir package (i.e., PY_DEST's parent) to be on the Python path.
        # For py-build-cmake builds, PY_DEST has no parent. For plain CMake
        # installs, PY_DEST's parent is lib/pythonX.Y.
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
    endif()

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
