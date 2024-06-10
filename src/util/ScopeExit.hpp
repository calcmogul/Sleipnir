// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

namespace sleipnir {

template <typename F>
class scope_exit {
 public:
  constexpr explicit scope_exit(F&& f) noexcept : m_f{std::forward<F>(f)} {}

  constexpr ~scope_exit() {
    if (m_active) {
      m_f();
    }
  }

  constexpr scope_exit(scope_exit&& rhs) noexcept
      : m_f{std::move(rhs.m_f)}, m_active{rhs.m_active} {
    rhs.release();
  }

  constexpr scope_exit(const scope_exit&) = delete;
  constexpr scope_exit& operator=(const scope_exit&) = delete;

  constexpr void release() noexcept { m_active = false; }

 private:
  F m_f;
  bool m_active = true;
};

}  // namespace sleipnir
