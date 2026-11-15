# Sleipnir

![C++](https://github.com/SleipnirGroup/Sleipnir/actions/workflows/cpp.yml/badge.svg)
![Python](https://github.com/SleipnirGroup/Sleipnir/actions/workflows/python.yml/badge.svg)
[![PyPI Downloads](https://img.shields.io/pypi/dm/sleipnirgroup-jormungandr.svg?label=PyPI%20Downloads)](https://pypi.org/project/sleipnirgroup-jormungandr/)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fsleipnirgroup.github.io%2FSleipnir%2F&label=Website)](https://sleipnirgroup.github.io/Sleipnir/)
[![C++ API](https://img.shields.io/badge/documentation-C%2B%2B-blue?label=API%20Docs)](https://sleipnirgroup.github.io/Sleipnir/docs/cpp)
[![Python API](https://img.shields.io/badge/documentation-Python-blue?label=API%20Docs)](https://sleipnirgroup.github.io/Sleipnir/docs/py)
[![Discord](https://img.shields.io/discord/975739302933856277?color=%23738ADB&label=Join%20our%20Discord&logo=discord&logoColor=white)](https://discord.gg/ad2EEZZwsS)

Sleipnir is a reverse mode autodiff library, interior-point method, and NLP solver DSL for C++23, Python, and Rust. The DSL automatically chooses the best solver based on the problem structure.

```cpp
#include <print>

#include <sleipnir/optimization/problem.hpp>

int main() {
  // Find the x, y pair with the largest product for which x + 3y = 36
  slp::Problem<double> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();

  problem.maximize(x * y);
  problem.subject_to(x + 3 * y == 36);
  problem.solve();

  // x = 18.0, y = 6.0
  std::println("x = {}, y = {}", x.value(), y.value());
}
```

```python
#!/usr/bin/env python

from sleipnir.optimization import Problem


def main():
    # Find the x, y pair with the largest product for which x + 3y = 36
    problem = Problem()

    x, y = problem.decision_variable(2)

    problem.maximize(x * y)
    problem.subject_to(x + 3 * y == 36)
    problem.solve()

    # x = 18.0, y = 6.0
    print(f"x = {x.value()}, y = {y.value()}")


if __name__ == "__main__":
    main()
```

Sleipnir supports easy cross-compilation via CMake toolchain files, provides good performance by default, and contains clean [reference implementations](https://github.com/SleipnirGroup/Sleipnir/tree/main/include/sleipnir/optimization/solver) of various solvers with links to explanatory material.

The C++ API also supports arbitrary scalar types, so users can specify higher precision floating-point types at the cost of speed.

## Install

The following platforms are supported:

* Windows
  * OS: Windows 11
  * Runtime: [Microsoft Visual C++ 2022 redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) from Visual Studio 2022 17.13
* Linux
  * OS: Ubuntu 24.04
  * Runtime: GCC 14 libstdc++ (run `sudo apt install g++-14`)
* macOS
  * OS: macOS 14.5
  * Runtime: Apple Clang 16.0.0 libc++ from Xcode 16.2 (run `xcode-select --install`)

To use Sleipnir within a CMake project, add the following to CMakeLists.txt:
```cmake
include(FetchContent)

FetchContent_Declare(
    Sleipnir
    GIT_REPOSITORY https://github.com/SleipnirGroup/Sleipnir.git
    GIT_TAG main
    EXCLUDE_FROM_ALL
    SYSTEM
)
FetchContent_MakeAvailable(Sleipnir)

target_link_libraries(MyApp PUBLIC Sleipnir::Sleipnir)
```

To use Sleipnir in Python, install `sleipnirgroup-jormungandr` from PyPI:
```bash
pip install sleipnirgroup-jormungandr
```

To use Sleipnir in Rust, add the `sleipnir` crate from this repository to Cargo.toml. It builds the C++ library from source, which also requires CMake 3.28 or newer and the toolchain listed above. See [its README](https://github.com/SleipnirGroup/Sleipnir/blob/main/rust/sleipnir/README.md) for details.
```toml
[dependencies]
sleipnir = { git = "https://github.com/SleipnirGroup/Sleipnir.git" }
```

If necessary, follow [these instructions](https://sleipnirgroup.github.io/Sleipnir/md_contributing.html) to build from source.

## Docs

See the [C++ API docs](https://sleipnirgroup.github.io/Sleipnir/docs/cpp), [Python API docs](https://sleipnirgroup.github.io/Sleipnir/docs/py), and [usage docs](https://sleipnirgroup.github.io/Sleipnir/md_usage.html).

## Examples

See the [examples folder](https://github.com/SleipnirGroup/Sleipnir/tree/main/examples), [C++ optimization unit tests](https://github.com/SleipnirGroup/Sleipnir/tree/main/test/src/optimization), and [Python optimization unit tests](https://github.com/SleipnirGroup/Sleipnir/tree/main/python/test/optimization).

## Benchmarks

See the [benchmarks folder](https://github.com/SleipnirGroup/Sleipnir/tree/main/benchmarks).

## Contributing

See the [contributing docs](https://sleipnirgroup.github.io/Sleipnir/md_contributing.html).
