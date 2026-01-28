// Copyright (c) Sleipnir contributors

#pragma once

namespace slp {

/// Iterates over a list of types and calls a function with each.
///
/// @tparam Ts The list of types.
/// @tparam F Callable type.
/// @param f The callable.
template <typename... Ts, typename F>
void for_each_type(F f) {
  (f.template operator()<Ts>(), ...);
}

}  // namespace slp
