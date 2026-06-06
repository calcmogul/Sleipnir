function(pybind11_mkdoc target headers)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Target compiler flags
    get_target_property(target_flags ${target} INCLUDE_DIRECTORIES)
    list(TRANSFORM target_flags PREPEND "-I")

    # Get default clang version
    execute_process(
        COMMAND clang++ --version
        OUTPUT_VARIABLE clang_version
        COMMAND_ERROR_IS_FATAL ANY
    )
    string(REGEX MATCH "[0-9]+" clang_version ${clang_version})

    # LLVM environment variables
    if(UNIX AND NOT APPLE)
        if(EXISTS /usr/lib/libclang.so)
            set(env_vars
                LLVM_DIR_PATH=/usr/lib
                LIBCLANG_PATH=/usr/lib/libclang.so
            )
        elseif(EXISTS /usr/lib/llvm-${clang_version}/lib/libclang.so)
            set(env_vars
                LLVM_DIR_PATH=/usr/lib/llvm-${clang_version}
                LIBCLANG_PATH=/usr/lib/llvm-${clang_version}/lib/libclang.so
            )
        endif()
    endif()

    # Clang compiler flags
    if(UNIX AND NOT APPLE)
        set(clang_flags -I/usr/lib/clang/${clang_version}/include)
    endif()

    # Eigen compiler flags
    get_target_property(eigen_flags Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    list(FILTER eigen_flags INCLUDE REGEX "\\$<BUILD_INTERFACE:.*>")
    list(TRANSFORM eigen_flags PREPEND "-I")

    # clang on Windows aarch64 failed to parse Eigen's NEON headers
    list(APPEND eigen_flags "-DEIGEN_DONT_VECTORIZE")

    # small_vector compiler flags
    get_target_property(
        small_vector_flags
        small_vector
        INTERFACE_INCLUDE_DIRECTORIES
    )
    list(FILTER small_vector_flags INCLUDE REGEX "\\$<BUILD_INTERFACE:.*>")
    list(TRANSFORM small_vector_flags PREPEND "-I")

    # TODO: Remove when Python 3.15 makes UTF-8 the default
    set(env_vars ${env_vars} PYTHONUTF8=1)

    # FastAD compiler flags
    get_target_property(fastad_flags FastAD INTERFACE_INCLUDE_DIRECTORIES)
    list(FILTER fastad_flags INCLUDE REGEX "\\$<BUILD_INTERFACE:.*>")
    list(TRANSFORM fastad_flags PREPEND "-I")

    # Generate docstrings.hpp
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp
        COMMAND
            ${CMAKE_COMMAND} -E env ${env_vars} ${Python3_EXECUTABLE} -m
            pybind11_mkdoc ${headers} -o
            ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp
            ${target_flags} ${clang_flags} ${eigen_flags} ${small_vector_flags}
            ${fastad_flags}
            -std=c++23
        DEPENDS ${headers}
    )

    # Add docstrings target
    add_custom_target(
        ${target}_docstrings
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp
    )
endfunction()
