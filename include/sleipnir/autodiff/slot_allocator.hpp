// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <gch/small_vector.hpp>

#include "sleipnir/util/flat_map.hpp"

namespace slp::detail {

/// Allocates blocks of contiguous tape slots.
///
/// A block is allocated when its value is first written and freed once the
/// value is dead. Freed blocks are reused by blocks of the same size, most
/// recently freed first, so the slots in use at any time stay few and
/// cache-resident.
class SlotAllocator {
 public:
  /// Allocates a block of slots, reusing a freed block of the same size if
  /// there is one.
  ///
  /// @param size The number of slots in the block.
  /// @return The block's first slot.
  uint32_t allocate(uint32_t size = 1) {
    auto& blocks = free_blocks(size);
    if (blocks.empty()) {
      return allocate_new(size);
    }

    const uint32_t first = blocks.back();
    blocks.pop_back();
    return first;
  }

  /// Allocates a block of slots that no other block has used, for values that
  /// must outlive the replay (e.g., results).
  ///
  /// @param size The number of slots in the block.
  /// @return The block's first slot.
  uint32_t allocate_new(uint32_t size = 1) {
    const uint32_t first = m_num_slots;
    m_num_slots += size;
    return first;
  }

  /// Frees a block of slots for reuse.
  ///
  /// @param first The block's first slot.
  /// @param size The number of slots in the block.
  void free(uint32_t first, uint32_t size = 1) {
    free_blocks(size).push_back(first);
  }

  /// Returns the number of slots allocated so far.
  ///
  /// @return The number of slots allocated so far.
  uint32_t size() const { return m_num_slots; }

 private:
  uint32_t m_num_slots = 0;

  /// The first slots of the free blocks with one slot, which most blocks
  /// have. They're kept out of m_free_blocks to skip its lookup.
  gch::small_vector<uint32_t> m_free_singles;

  /// Maps a block size other than 1 to the first slots of the free blocks of
  /// that size.
  flat_map<uint32_t, gch::small_vector<uint32_t>> m_free_blocks;

  /// Returns the first slots of the free blocks of the given size.
  gch::small_vector<uint32_t>& free_blocks(uint32_t size) {
    return size == 1 ? m_free_singles : m_free_blocks[size];
  }
};

}  // namespace slp::detail
