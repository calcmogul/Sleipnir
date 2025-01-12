// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <fstream>
#include <string>
#include <string_view>

#include <Eigen/SparseCore>

#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * Writes the sparsity pattern of a sparse matrix to a file.
 *
 * Each file represents the sparsity pattern of one matrix over time. <a
 * href="https://github.com/SleipnirGroup/Sleipnir/blob/main/tools/spy.py">spy.py</a>
 * can display it as an animation.
 *
 * The file starts with the following header:
 * <ol>
 *   <li>Plot title (length as a little-endian int32, then characters)</li>
 *   <li>Row label (length as a little-endian int32, then characters)</li>
 *   <li>Column label (length as a little-endian int32, then characters)</li>
 * </ol>
 *
 * Then, each sparsity pattern starts with:
 * <ol>
 *   <li>Number of coordinates as a little-endian int32</li>
 * </ol>
 *
 * followed by that many coordinates in the following format:
 * <ol>
 *   <li>Row index as a little-endian int32</li>
 *   <li>Column index as a little-endian int32</li>
 *   <li>Sign as a character ('+' for positive, '-' for negative, or '0' for
 *       zero)</li>
 * </ol>
 *
 * @param[out] file A file stream.
 * @param[in] mat The sparse matrix.
 */
class SLEIPNIR_DLLEXPORT Spy {
 public:
  /**
   * Constructs a Spy instance.
   *
   * @param filename The filename.
   * @param title Plot title.
   * @param rowLabel Row label.
   * @param colLabel Column label.
   * @param rows The sparse matrix's number of rows.
   * @param cols The sparse matrix's number of columns.
   */
  Spy(std::string_view filename, std::string_view title,
      std::string_view rowLabel, std::string_view colLabel, int rows, int cols)
      : m_file{std::string{filename}, std::ios::binary} {
    // Write title
    Write32le(title.size());
    m_file.write(title.data(), title.size());

    // Write row label
    Write32le(rowLabel.size());
    m_file.write(rowLabel.data(), rowLabel.size());

    // Write column label
    Write32le(colLabel.size());
    m_file.write(colLabel.data(), colLabel.size());

    // Write row and column counts
    Write32le(rows);
    Write32le(cols);
  }

  /**
   * Adds a matrix to the file.
   *
   * @param mat The matrix.
   */
  void Add(const Eigen::SparseMatrix<double>& mat) {
    // Write number of coordinates
    Write32le(mat.nonZeros());

    // Write coordinates
    for (int k = 0; k < mat.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it{mat, k}; it; ++it) {
        Write32le(it.row());
        Write32le(it.col());
        if (it.value() > 0.0) {
          m_file << '+';
        } else if (it.value() < 0.0) {
          m_file << '-';
        } else {
          m_file << '0';
        }
      }
    }
  }

 private:
  std::ofstream m_file;

  /**
   * Writes a 32-bit signed integer to the file as little-endian.
   *
   * @param num A 32-bit signed integer.
   */
  void Write32le(int32_t num) {
    if constexpr (std::endian::native != std::endian::little) {
      auto value_representation =
          std::bit_cast<std::array<std::byte, sizeof(int32_t)>>(num);
      std::ranges::reverse(value_representation);
      num = std::bit_cast<int32_t>(value_representation);
    }
    m_file.write(reinterpret_cast<char*>(&num), sizeof(num));
  }
};

}  // namespace sleipnir
