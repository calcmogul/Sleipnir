// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

namespace wpi::util {

template <typename T>
T byteswap(T n) noexcept {
  auto value_representation =
      std::bit_cast<std::array<std::byte, sizeof(int32_t)>>(n);
  std::ranges::reverse(value_representation);
  return std::bit_cast<int32_t>(value_representation);
}

}  // namespace wpi::util
