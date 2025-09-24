// Copyright (c) Sleipnir contributors

#pragma once

/// Forces a function to be inlined at its call sites.
///
/// This is for functions on hot paths that the compiler's inlining heuristics
/// consider too large to inline, but that only have one or two call sites
/// each.
#if defined(__GNUC__)
#define SLEIPNIR_FORCE_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
#define SLEIPNIR_FORCE_INLINE __forceinline
#else
#define SLEIPNIR_FORCE_INLINE inline
#endif
