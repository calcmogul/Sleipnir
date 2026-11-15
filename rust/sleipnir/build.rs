// Copyright (c) Sleipnir contributors

//! Builds the C++ library with CMake and compiles the cxx bridge against it.
//!
//! The C++ sources come from, in order:
//!
//! 1. The `SLEIPNIR_SOURCE_DIR` environment variable
//! 2. The repository checkout containing this crate
//! 3. `cpp-src/` in this crate, which `vendor_cpp.py` populates before
//!    publishing
//!
//! CMake fetches Eigen and small_vector. To build offline, point CMake at
//! existing checkouts with the `FETCHCONTENT_SOURCE_DIR_EIGEN3` and
//! `FETCHCONTENT_SOURCE_DIR_SMALL_VECTOR` environment variables; every
//! `FETCHCONTENT_*` environment variable is forwarded as a CMake cache
//! variable.

use std::env;
use std::path::{Path, PathBuf};

const ENV_VARS: [&str; 4] = [
    "SLEIPNIR_SOURCE_DIR",
    "FETCHCONTENT_SOURCE_DIR_EIGEN3",
    "FETCHCONTENT_SOURCE_DIR_SMALL_VECTOR",
    "FETCHCONTENT_FULLY_DISCONNECTED",
];

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source_dir = cpp_source_dir(&manifest_dir);

    for path in ["CMakeLists.txt", "cmake", "include", "src"] {
        println!("cargo:rerun-if-changed={}", source_dir.join(path).display());
    }
    for path in ["cpp", "src/ffi.rs"] {
        println!(
            "cargo:rerun-if-changed={}",
            manifest_dir.join(path).display()
        );
    }
    for var in ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }

    // Build only the library target; the install rules would also install
    // the dependencies
    let mut config = cmake::Config::new(&source_dir);
    config
        .profile("Release")
        .build_target("Sleipnir")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        config.cxxflag("/EHsc");
    }
    for (key, value) in env::vars().filter(|(key, _)| key.starts_with("FETCHCONTENT_")) {
        config.define(key, value);
    }
    let build_dir = config.build().join("build");

    let eigen_dir = fetched_source_dir("FETCHCONTENT_SOURCE_DIR_EIGEN3", &build_dir, "eigen3-src");
    let small_vector_dir = fetched_source_dir(
        "FETCHCONTENT_SOURCE_DIR_SMALL_VECTOR",
        &build_dir,
        "small_vector-src",
    );

    let mut bridge = cxx_build::bridge("src/ffi.rs");
    bridge
        .file("cpp/bridge.cpp")
        // The bridge instantiates the float solver from headers, which is too
        // slow unoptimized
        .opt_level(2)
        .define("NDEBUG", None)
        .include(source_dir.join("include"));
    let dependency_includes = [eigen_dir, small_vector_dir.join("source/include")];
    if bridge.get_compiler().is_like_msvc() {
        bridge
            .flag_if_supported("/std:c++23preview")
            .flag_if_supported("/std:c++23")
            .flag("/EHsc")
            .flag("/bigobj")
            .flag("/utf-8")
            .flag("/permissive-");
        for dir in &dependency_includes {
            bridge.include(dir);
        }
    } else {
        bridge.std("c++23");
        // System includes silence warnings from dependency headers
        for dir in &dependency_includes {
            bridge.flag("-isystem").flag(dir.to_str().unwrap());
        }
    }
    bridge.compile("sleipnir-bridge");

    // Link after the bridge so the static linker resolves its references.
    // Multi-config generators (e.g., Visual Studio) put the library in a
    // per-configuration subdirectory.
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("Release").display()
    );
    println!("cargo:rustc-link-lib=static=Sleipnir");
}

/// Returns the C++ library's source directory.
fn cpp_source_dir(manifest_dir: &Path) -> PathBuf {
    if let Some(dir) = env::var_os("SLEIPNIR_SOURCE_DIR") {
        return PathBuf::from(dir);
    }

    let is_source_dir =
        |dir: &Path| dir.join("CMakeLists.txt").is_file() && dir.join("include/sleipnir").is_dir();
    let repository = manifest_dir.parent().unwrap().parent().unwrap();
    let vendored = manifest_dir.join("cpp-src");
    if is_source_dir(repository) {
        repository.to_path_buf()
    } else if is_source_dir(&vendored) {
        vendored
    } else {
        panic!(
            "Sleipnir's C++ sources weren't found. Set SLEIPNIR_SOURCE_DIR, build from the \
             repository, or run vendor_cpp.py."
        );
    }
}

/// Returns a FetchContent dependency's source directory: the
/// `FETCHCONTENT_SOURCE_DIR_*` override in the CMake cache if set, or the
/// default download location otherwise.
fn fetched_source_dir(var: &str, build_dir: &Path, default: &str) -> PathBuf {
    let cache = std::fs::read_to_string(build_dir.join("CMakeCache.txt")).unwrap();
    cache
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.split(':').next() == Some(var) && !value.is_empty()).then(|| PathBuf::from(value))
        })
        .unwrap_or_else(|| build_dir.join("_deps").join(default))
}
