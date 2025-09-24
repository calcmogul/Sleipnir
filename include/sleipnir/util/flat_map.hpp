// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <compare>
#include <concepts>
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace slp {

template <class KeyRef, class TRef, class KeyIter, class MappedIter>
  requires std::is_reference_v<KeyRef> && std::is_reference_v<TRef>
struct flat_map_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type =
      std::pair<std::remove_cvref_t<KeyRef>, std::remove_cvref_t<TRef>>;
  using difference_type =
      typename std::iterator_traits<KeyIter>::difference_type;
  using reference = std::pair<KeyRef, TRef>;

  struct pointer {
    constexpr explicit pointer(reference value) noexcept
        : value{std::move(value)} {}

    constexpr reference* operator->() noexcept { return std::addressof(value); }

    constexpr const reference* operator->() const noexcept {
      return std::addressof(value);
    }

   private:
    reference value;
  };

  constexpr flat_map_iterator() = default;

  constexpr flat_map_iterator(KeyIter key_it, MappedIter mapped_it)
      : key_it{key_it}, mapped_it{mapped_it} {}

  constexpr reference operator*() const noexcept {
    return reference{*key_it, *mapped_it};
  }

  constexpr pointer operator->() const noexcept {
    return pointer{reference{*key_it, *mapped_it}};
  }

  constexpr reference operator[](difference_type n) const noexcept {
    return reference{*(key_it + n), *(mapped_it + n)};
  }

  constexpr flat_map_iterator operator+(difference_type n) const noexcept {
    return flat_map_iterator{key_it + n, mapped_it + n};
  }

  constexpr flat_map_iterator operator-(difference_type n) const noexcept {
    return flat_map_iterator{key_it - n, mapped_it - n};
  }

  constexpr flat_map_iterator& operator++() noexcept {
    ++key_it;
    ++mapped_it;
    return *this;
  }

  constexpr flat_map_iterator operator++(int) noexcept {
    flat_map_iterator tmp{*this};
    ++key_it;
    ++mapped_it;
    return tmp;
  }

  constexpr flat_map_iterator& operator--() noexcept {
    --key_it;
    --mapped_it;
    return *this;
  }

  constexpr flat_map_iterator operator--(int) noexcept {
    flat_map_iterator tmp{*this};
    --key_it;
    --mapped_it;
    return tmp;
  }

  constexpr flat_map_iterator& operator+=(difference_type n) noexcept {
    key_it += n;
    mapped_it += n;
    return *this;
  }

  constexpr flat_map_iterator& operator-=(difference_type n) noexcept {
    key_it -= n;
    mapped_it -= n;
    return *this;
  }

  constexpr KeyIter key_iter() const { return key_it; }

  constexpr MappedIter mapped_iter() const { return mapped_it; }

  friend constexpr std::strong_ordering operator<=>(
      const flat_map_iterator& lhs, const flat_map_iterator& rhs) {
    return lhs.key_it <=> rhs.key_it;
  }

  friend constexpr bool operator!=(const flat_map_iterator& lhs,
                                   const flat_map_iterator& rhs) {
    return lhs.key_it != rhs.key_it;
  }

 private:
  KeyIter key_it;
  MappedIter mapped_it;
};

template <class Key, class T, class Compare = std::less<Key>,
          class KeyContainer = std::vector<Key>,
          class MappedContainer = std::vector<T>>
class flat_map {
 public:
  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<key_type, mapped_type>;
  using reference = std::pair<const key_type&, mapped_type&>;
  using const_reference = std::pair<const key_type&, const mapped_type&>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using iterator = flat_map_iterator<const key_type&, mapped_type&,
                                     typename KeyContainer::const_iterator,
                                     typename MappedContainer::iterator>;
  using const_iterator =
      flat_map_iterator<const key_type&, const mapped_type&,
                        typename KeyContainer::const_iterator,
                        typename MappedContainer::const_iterator>;
  using reverse_iterator =
      flat_map_iterator<const key_type&, mapped_type&,
                        typename KeyContainer::const_reverse_iterator,
                        typename MappedContainer::reverse_iterator>;
  using const_reverse_iterator =
      flat_map_iterator<const key_type&, const mapped_type&,
                        typename KeyContainer::const_reverse_iterator,
                        typename MappedContainer::const_reverse_iterator>;

  // Element access

  constexpr mapped_type& operator[](const key_type& x) {
    return try_emplace(x).first->second;
  }

  constexpr mapped_type& operator[](key_type&& x) {
    return try_emplace(std::move(x)).first->second;
  }

  // Iterators

  constexpr iterator begin() noexcept {
    return iterator{c.keys.begin(), c.values.begin()};
  }

  constexpr const_iterator begin() const noexcept {
    return const_iterator{c.keys.begin(), c.values.begin()};
  }

  constexpr iterator end() noexcept {
    return iterator{c.keys.end(), c.values.end()};
  }

  constexpr const_iterator end() const noexcept {
    return const_iterator{c.keys.end(), c.values.end()};
  }

  constexpr reverse_iterator rbegin() noexcept {
    return reverse_iterator{c.keys.rbegin(), c.values.rbegin()};
  }

  constexpr const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator{c.keys.rbegin(), c.values.rbegin()};
  }

  constexpr reverse_iterator rend() noexcept {
    return reverse_iterator{c.keys.rend(), c.values.rend()};
  }

  constexpr const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator{c.keys.rend(), c.values.rend()};
  }

  constexpr const_iterator cbegin() const noexcept { return begin(); }

  constexpr const_iterator cend() const noexcept { return end(); }

  constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }

  constexpr const_reverse_iterator crend() const noexcept { return rend(); }

  // Capacity

  [[nodiscard]]
  constexpr bool empty() const noexcept {
    return c.keys.empty();
  }

  constexpr size_type size() const noexcept { return c.keys.size(); }

  // Modifiers

  template <class... Args>
    requires std::constructible_from<mapped_type, Args&&...>
  constexpr std::pair<iterator, bool> try_emplace(const key_type& k,
                                                  Args&&... args) {
    auto it = std::ranges::lower_bound(c.keys, k, compare);
    if (it == c.keys.end() || compare(*it, k) || compare(k, *it)) {
      auto values_it =
          c.values.emplace(project(it), std::forward<Args>(args)...);
      it = c.keys.insert(it, k);
      return {iterator{it, values_it}, true};
    }
    return {iterator{it, project(it)}, false};
  }

  template <class... Args>
    requires std::constructible_from<mapped_type, Args&&...>
  constexpr std::pair<iterator, bool> try_emplace(key_type&& k,
                                                  Args&&... args) {
    auto it = std::ranges::lower_bound(c.keys, k, compare);
    if (it == c.keys.end() || compare(*it, k) || compare(k, *it)) {
      auto values_it =
          c.values.emplace(project(it), std::forward<Args>(args)...);
      it = c.keys.insert(it, std::forward<key_type>(k));
      return {iterator{it, values_it}, true};
    }
    return {iterator{it, project(it)}, false};
  }

  constexpr iterator erase(iterator position) {
    return iterator{c.keys.erase(position.key_iter()),
                    c.values.erase(position.mapped_iter())};
  }

  constexpr iterator erase(const_iterator position) {
    return iterator{c.keys.erase(position.key_iter()),
                    c.values.erase(position.mapped_iter())};
  }

  constexpr void clear() noexcept {
    c.keys.clear();
    c.values.clear();
  }

  // Lookup

  constexpr iterator find(const key_type& x) {
    auto it = key_find(x);
    return iterator{it, project(it)};
  }

  constexpr const_iterator find(const key_type& x) const {
    auto it = key_find(x);
    return const_iterator{it, project(it)};
  }

  template <class K>
  constexpr iterator find(const K& x) {
    auto it = key_find(x);
    return iterator{it, project(it)};
  }

  template <class K>
  constexpr const_iterator find(const K& x) const {
    auto it = key_find(x);
    return iterator{it, project(it)};
  }

  constexpr bool contains(const key_type& x) const {
    return key_find(x) != c.keys.end();
  }

  template <class K>
  constexpr bool contains(const K& x) const {
    return key_find(x) != c.keys.end();
  }

  // Observers

  constexpr const KeyContainer& keys() const noexcept { return c.keys; }

  constexpr const MappedContainer& values() const noexcept { return c.values; }

  // Comparison

  friend constexpr bool operator==(const flat_map& x, const flat_map& y) {
    return std::ranges::equal(x, y);
  }

 private:
  struct Containers {
    KeyContainer keys;
    MappedContainer values;
  };

  Containers c;
  Compare compare;

  using key_iter_t = typename KeyContainer::iterator;
  using key_const_iter_t = typename KeyContainer::const_iterator;
  using mapped_iter_t = typename MappedContainer::iterator;
  using mapped_const_iter_t = typename MappedContainer::const_iterator;

  constexpr typename MappedContainer::iterator project(
      typename KeyContainer::iterator key_it) {
    return c.values.begin() + (key_it - c.keys.begin());
  }

  constexpr typename MappedContainer::const_iterator project(
      typename KeyContainer::const_iterator key_it) const {
    return c.values.begin() + (key_it - c.keys.begin());
  }

  template <typename K>
  constexpr typename KeyContainer::iterator key_find(const K& k) {
    auto it = std::ranges::lower_bound(c.keys, k, compare);
    if (it != c.keys.end() && (compare(*it, k) || compare(k, *it))) {
      return c.keys.end();
    } else {
      return it;
    }
  }

  template <typename K>
  constexpr typename KeyContainer::const_iterator key_find(const K& k) const {
    auto it = std::ranges::lower_bound(c.keys, k, compare);
    if (it != c.keys.end() && (compare(*it, k) || compare(k, *it))) {
      return c.keys.end();
    } else {
      return it;
    }
  }
};

}  // namespace slp
