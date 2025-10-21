// Copyright (c) Sleipnir contributors

#pragma once

template <typename Scalar>
Scalar lerp(Scalar a, Scalar b, Scalar t) {
  return a + (b - a) * t;
}
