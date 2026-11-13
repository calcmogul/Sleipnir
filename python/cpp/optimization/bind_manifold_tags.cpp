// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>

#include <sleipnir/optimization/manifold.hpp>

namespace nb = nanobind;

namespace slp {

void bind_manifold_tags(nb::module_& optimization) {
  // The tag types have no Python constructor, so these are their only
  // instances
  optimization.attr("sphere") = nb::cast(sphere);
  optimization.attr("so2") = nb::cast(so2);
  optimization.attr("so3") = nb::cast(so3);
  optimization.attr("se2") = nb::cast(se2);
  optimization.attr("se3") = nb::cast(se3);
}

}  // namespace slp
