# Disable in-source builds to prevent source tree corruption
if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
    message(
        FATAL_ERROR
        "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
"
    )
endif()

cmake_minimum_required(VERSION 3.21)
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
)

# Make `GIT_SUBMODULES ""` initialize no submodules
cmake_policy(SET CMP0097 NEW)

# Set default build type to release with debug info (i.e. release mode
# optimizations are performed, but debug info still exists).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

project(Sleipnir LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# When building, don't use the install RPATH already (but later on when
# installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# The RPATH to be used when installing, but only if it's not a system directory
list(
    FIND
    CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
    "${CMAKE_INSTALL_PREFIX}/lib"
    isSystemDir
)
if("${isSystemDir}" STREQUAL "-1")
    list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# Generate compile_commands.json by default
if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS "YES" CACHE STRING "" FORCE)
endif()

include(BuildTypes)

# Control where the static and shared libraries are built so that on Windows,
# we don't need to tinker with the path to run the executable
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS FALSE)

option(BUILD_BENCHMARKS "Build CasADi and Sleipnir benchmarks" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_PYTHON "Build Python module" OFF)
option(DISABLE_DIAGNOSTICS "Disable diagnostics support at compile-time" OFF)

include(CompilerFlags)

file(GLOB_RECURSE Sleipnir_src src/*.cpp)
if(TARGET_SUPPORTS_SHARED_LIBS)
    add_library(Sleipnir ${Sleipnir_src})
else()
    add_library(Sleipnir STATIC ${Sleipnir_src})
endif()
add_library(Sleipnir::Sleipnir ALIAS Sleipnir)
compiler_flags(Sleipnir)
target_include_directories(Sleipnir PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set_target_properties(Sleipnir PROPERTIES DEBUG_POSTFIX "d")

set_property(TARGET Sleipnir PROPERTY FOLDER "libraries")
target_compile_definitions(Sleipnir PRIVATE SLEIPNIR_EXPORTS)

include(CTest)
include(FetchContent)

# Options for using a package manager (e.g., vcpkg) for certain dependencies
option(USE_SYSTEM_EIGEN "Use system eigen" OFF)
option(USE_SYSTEM_NANOBIND "Use system nanobind" OFF)

# Required for std::async()
find_package(Threads)
if(Threads_FOUND)
    target_link_libraries(Sleipnir PUBLIC Threads::Threads)
endif()

if(DISABLE_DIAGNOSTICS)
    target_compile_definitions(Sleipnir PUBLIC SLEIPNIR_DISABLE_DIAGNOSTICS)
endif()

# Eigen dependency
if(NOT USE_SYSTEM_EIGEN)
    set(EIGEN_BUILD_CMAKE_PACKAGE TRUE)
    fetchcontent_declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        # master on 2025-01-02
        GIT_TAG 7bb23b1e360e22a395eef75061565fc079ae6a85
    )
    fetchcontent_makeavailable(Eigen3)
else()
    find_package(Eigen3 CONFIG REQUIRED)
endif()

target_link_libraries(Sleipnir PUBLIC Eigen3::Eigen)

# fmt dependency
fetchcontent_declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.1.2
)
fetchcontent_makeavailable(fmt)

target_link_libraries(Sleipnir PUBLIC fmt::fmt)

if(BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
    # Catch2 dependency
    fetchcontent_declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
        CMAKE_ARGS
    )
    fetchcontent_makeavailable(Catch2)
endif()

target_include_directories(
    Sleipnir
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

install(
    TARGETS Sleipnir
    COMPONENT Sleipnir
    EXPORT SleipnirTargets
    LIBRARY
    DESTINATION lib
    ARCHIVE
    DESTINATION lib
    RUNTIME
    DESTINATION bin
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

include(CMakePackageConfigHelpers)

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
    foreach(benchmark "CartPole" "Flywheel")
        file(
            GLOB ${benchmark}PerfBenchmark_src
            benchmarks/perf/*.cpp
            benchmarks/perf/${benchmark}/*.cpp
        )
        add_executable(
            ${benchmark}PerfBenchmark
            ${${benchmark}PerfBenchmark_src}
        )
        compiler_flags(${benchmark}PerfBenchmark)
        target_include_directories(
            ${benchmark}PerfBenchmark
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
                ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/perf
        )
        target_link_libraries(${benchmark}PerfBenchmark PRIVATE Sleipnir)
    endforeach()

    # Scalability benchmark (if CasADi exists)
    find_package(casadi QUIET)
    if(casadi_FOUND)
        foreach(benchmark "CartPole" "Flywheel")
            file(
                GLOB ${benchmark}ScalabilityBenchmark_src
                benchmarks/scalability/*.cpp
                benchmarks/scalability/${benchmark}/*.cpp
            )
            add_executable(
                ${benchmark}ScalabilityBenchmark
                ${${benchmark}ScalabilityBenchmark_src}
            )
            compiler_flags(${benchmark}ScalabilityBenchmark)
            target_include_directories(
                ${benchmark}ScalabilityBenchmark
                PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
                    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/scalability
            )
            target_link_libraries(
                ${benchmark}ScalabilityBenchmark
                PRIVATE Sleipnir casadi
            )
        endforeach()
    endif()
endif()

if(BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
    enable_testing()
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    include(Catch)

    # Build Sleipnir tests
    file(GLOB_RECURSE Sleipnir_test_src test/src/*.cpp)
    add_executable(SleipnirTest ${Sleipnir_test_src})
    compiler_flags(SleipnirTest)
    target_include_directories(
        SleipnirTest
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    )
    target_link_libraries(SleipnirTest PRIVATE Sleipnir Catch2::Catch2WithMain)
    if(NOT CMAKE_TOOLCHAIN_FILE)
        catch_discover_tests(SleipnirTest)
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
        target_link_libraries(${example} PRIVATE Sleipnir)

        # Build example test if files exist for it
        if(
            BUILD_TESTING
            AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/test
        )
            file(GLOB_RECURSE test_sources examples/${example}/test/*.cpp)
            add_executable(${example}Test ${sources} ${test_sources})
            compiler_flags(${example}Test)
            target_compile_definitions(${example}Test PUBLIC RUNNING_TESTS)
            target_include_directories(
                ${example}Test
                PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/test/include
            )
            target_link_libraries(
                ${example}Test
                PRIVATE Sleipnir Catch2::Catch2WithMain
            )
            if(NOT CMAKE_TOOLCHAIN_FILE)
                catch_discover_tests(${example}Test)
            endif()
        endif()
    endforeach()
endif()

if(BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    if(DEFINED PY_BUILD_CMAKE_MODULE_NAME)
        set(PY_DEST ${PY_BUILD_CMAKE_MODULE_NAME})
    else()
        set(PY_DEST lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
    endif()

    # nanobind dependency
    if(NOT USE_SYSTEM_NANOBIND)
        fetchcontent_declare(
            nanobind
            GIT_REPOSITORY https://github.com/wjakob/nanobind.git
            GIT_TAG v2.4.0
        )
        fetchcontent_makeavailable(nanobind)
    else()
        find_package(nanobind CONFIG REQUIRED)
    endif()

    file(GLOB_RECURSE jormungandr_src jormungandr/cpp/*.cpp)

    # Build Sleipnir dependency directly into the wheel to avoid having to
    # configure RPATHs
    nanobind_add_module(_jormungandr ${jormungandr_src} ${Sleipnir_src})
    compiler_flags(_jormungandr)
    target_compile_definitions(_jormungandr PRIVATE JORMUNGANDR=1)
    target_include_directories(
        _jormungandr
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/jormungandr/cpp
    )
    target_link_libraries(_jormungandr PUBLIC Threads::Threads Eigen3::Eigen)

    # Suppress compiler warnings in nanobind
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        # nanobind/include/nanobind/nb_attr.h:212:14: warning: ISO C++ forbids zero-size array
        #  212 |     arg_data args[Size];
        #      |              ^~~~
        target_compile_options(_jormungandr PRIVATE -Wno-pedantic)
    endif()
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
        target_compile_options(nanobind-static PRIVATE -Wno-array-bounds)
        target_compile_options(
            _jormungandr
            PRIVATE -Wno-nested-anon-types -Wno-zero-length-array
        )
    endif()

    install(
        TARGETS _jormungandr
        COMPONENT python_modules
        LIBRARY
        DESTINATION ${PY_DEST}
    )

    nanobind_add_stub(
        _jormungandr_stub
        INSTALL_TIME
        MARKER_FILE jormungandr/py.typed
        MODULE _jormungandr
        OUTPUT jormungandr/__init__.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:_jormungandr>
        DEPENDS _jormungandr
        COMPONENT python_modules
    )
    nanobind_add_stub(
        _jormungandr_autodiff_stub
        INSTALL_TIME
        MODULE _jormungandr.autodiff
        OUTPUT jormungandr/autodiff/__init__.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:_jormungandr>
        DEPENDS _jormungandr
        COMPONENT python_modules
    )
    nanobind_add_stub(
        _jormungandr_optimization_stub
        INSTALL_TIME
        MODULE _jormungandr.optimization
        OUTPUT jormungandr/optimization/__init__.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:_jormungandr>
        DEPENDS _jormungandr
        COMPONENT python_modules
    )

    # pybind11_mkdoc doesn't support Windows
    if(NOT WIN32 AND NOT CMAKE_CROSSCOMPILING)
        # pybind11_mkdoc dependency
        fetchcontent_declare(
            pybind11_mkdoc
            GIT_REPOSITORY https://github.com/pybind/pybind11_mkdoc.git
            # master on 2023-02-08
            GIT_TAG 42fbf377824185e255b06d68fa70f4efcd569e2d
            GIT_SUBMODULES ""
        )
        fetchcontent_makeavailable(pybind11_mkdoc)

        file(
            GLOB_RECURSE sleipnir_headers
            include/sleipnir/autodiff/*.hpp
            include/sleipnir/control/*.hpp
            include/sleipnir/optimization/*.hpp
        )

        # Generate docs for the Python module
        include(cmake/modules/Pybind11Mkdoc.cmake)
        pybind11_mkdoc(_jormungandr "${sleipnir_headers}")
        add_dependencies(_jormungandr _jormungandr_docstrings)
    endif()
endif()
