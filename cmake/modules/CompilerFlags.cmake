macro(compiler_flags target)
    if(NOT MSVC)
        target_compile_options(
            ${target}
            PRIVATE -Wall -pedantic -Wextra -Werror -Wno-psabi
        )
    else()
        # Suppress the following warnings:
        #   * C4244: lossy conversion
        #   * C4251: missing dllexport/dllimport attribute on data member
        target_compile_options(${target} PRIVATE /wd4244 /wd4251 /WX)
    endif()

    # Disable warning false positives in Eigen
    if(
        ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU"
        AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "12"
    )
        target_compile_options(
            ${target}
            PRIVATE -Wno-array-bounds -Wno-stringop-overflow
        )
    endif()

    if(MSVC)
        target_compile_options(
            ${target}
            PUBLIC /std:c++23preview /MP /utf-8 /bigobj
        )
    elseif(APPLE)
        target_compile_options(${target} PUBLIC -Wno-pre-c++2b-compat)
    endif()

    if(NOT MSVC)
        target_compile_features(${target} PUBLIC cxx_std_23)
    endif()
endmacro()
