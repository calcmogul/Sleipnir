// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <span>
#include <type_traits>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

namespace slp::detail {

/// A sparse matrix entry whose value is read from a tape slot.
struct TapeEntry {
  /// The slot the entry's value is read from.
  uint32_t slot;

  /// The entry's row.
  int row;

  /// The entry's column.
  int col;
};

/// Maps a tape slot to an index in a sparse matrix's value array.
struct SlotMapEntry {
  /// The index of the slot the value is read from.
  uint32_t slot_idx;

  /// The index in the sparse matrix's value array.
  uint32_t value_idx;
};

/// Sets a sparse matrix's sparsity pattern to the given entries and returns
/// where in its value array each entry is stored.
///
/// @tparam Scalar Scalar type.
/// @param mat The sparse matrix.
/// @param entries The sparse matrix entries.
/// @return The slot map entry for each entry, in the same order as entries.
template <typename Scalar>
gch::small_vector<SlotMapEntry> make_slot_map(
    Eigen::SparseMatrix<Scalar>& mat, std::span<const TapeEntry> entries) {
  // Set sparsity pattern
  gch::small_vector<Eigen::Triplet<Scalar>> triplets;
  for (const auto& entry : entries) {
    triplets.emplace_back(entry.row, entry.col, Scalar(0));
  }
  mat.setFromTriplets(triplets.begin(), triplets.end());

  // Generate map from tape slot to value array position
  gch::small_vector<SlotMapEntry> slot_map;
  for (const auto& entry : entries) {
    slot_map.emplace_back(
        entry.slot, static_cast<uint32_t>(&mat.coeffRef(entry.row, entry.col) -
                                          mat.valuePtr()));
  }
  return slot_map;
}

/// Writes slot values into a sparse matrix's value array.
///
/// @tparam Scalar Scalar type.
/// @param m The sparse matrix.
/// @param slot_map Map from slot to value array index.
/// @param slots The slot values.
template <typename Scalar>
void copy_slots(Eigen::SparseMatrix<Scalar>& m,
                std::span<const SlotMapEntry> slot_map,
                std::type_identity_t<std::span<const Scalar>> slots) {
  auto values = m.valuePtr();
  for (const auto& entry : slot_map) {
    values[entry.value_idx] = slots[entry.slot_idx];
  }
}

}  // namespace slp::detail
