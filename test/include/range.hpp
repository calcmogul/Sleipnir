// Copyright (c) Sleipnir contributors

#pragma once

#include <ranges>

template <typename T>
auto range(T start, T end, T step) {
  return std::views::iota(0, static_cast<int>((end - start) / step)) |
         std::views::transform([=](auto&& i) { return start + i * step; });
}
