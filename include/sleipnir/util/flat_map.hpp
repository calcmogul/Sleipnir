// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <ranges>
#include <utility>
#include <vector>

namespace slp {

template <typename _T1, typename _T2>
struct ref_pair {
  static_assert(std::is_reference<_T1>{} && std::is_reference<_T2>{});

  using pair_type =
      std::pair<std::remove_cvref_t<_T1>, std::remove_cvref_t<_T2>>;
  using pair_of_references_type =
      std::pair<std::remove_cvref_t<_T1>&, std::remove_cvref_t<_T2>&>;
  using const_pair_of_references_type =
      std::pair<std::remove_cvref_t<_T1> const&,
                std::remove_cvref_t<_T2> const&>;

  ref_pair(_T1 t1, _T2 t2) : first(t1), second(t2) {}
  ref_pair(ref_pair const& other) : first(other.first), second(other.second) {}
  ref_pair(ref_pair&& other) : first(other.first), second(other.second) {}
  ref_pair const& operator=(ref_pair const& other) const {
    first = other.first;
    second = other.second;
    return *this;
  }
  ref_pair const& operator=(ref_pair&& other) const {
    first = other.first;
    second = other.second;
    return *this;
  }

  ref_pair const& operator=(pair_type const& other) const {
    first = other.first;
    second = other.second;
    return *this;
  }
  ref_pair const& operator=(pair_type&& other) const {
    first = std::move(other.first);
    second = std::move(other.second);
    return *this;
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator pair_type() const { return pair_type(first, second); }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator pair_of_references_type() const {
    return pair_of_references_type(first, second);
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator const_pair_of_references_type() const {
    return const_pair_of_references_type(first, second);
  }
  bool operator==(ref_pair rhs) const {
    return first == rhs.first && second == rhs.second;
  }
  bool operator!=(ref_pair rhs) const { return !(*this == rhs); }
  bool operator<(ref_pair rhs) const {
    if (first < rhs.first) {
      return true;
    }
    if (rhs.first < first) {
      return false;
    }
    return second < rhs.second;
  }

  _T1 first;
  _T2 second;
};

template <class _KeyRef, class _TRef, class _KeyIter, class _MappedIter>
struct flat_map_iterator {
  static_assert(std::is_reference<_KeyRef>{} && std::is_reference<_TRef>{});

  using iterator_category = std::random_access_iterator_tag;
  using value_type =
      std::pair<std::remove_cvref_t<_KeyRef>, std::remove_cvref_t<_TRef>>;
  using difference_type =
      typename std::iterator_traits<_KeyIter>::difference_type;
  using reference = ref_pair<_KeyRef, _TRef>;

  struct arrow_proxy {
    reference* operator->() noexcept { return &value_; }
    reference const* operator->() const noexcept { return &value_; }
    explicit arrow_proxy(reference value) noexcept : value_(std::move(value)) {}

   private:
    reference value_;
  };
  using pointer = arrow_proxy;

  flat_map_iterator() {}
  flat_map_iterator(_KeyIter key_it, _MappedIter mapped_it)
      : key_it_(key_it), mapped_it_(mapped_it) {}
  template <class _TRef2, class _MappedIter2>
  // NOLINTNEXTLINE(google-explicit-constructor)
  flat_map_iterator(
      flat_map_iterator<_KeyRef, _TRef2, _KeyIter, _MappedIter2> other)
    requires std::is_convertible_v<_TRef2, _TRef> &&
                 std::is_convertible_v<_MappedIter2, _MappedIter>
      : key_it_(other.key_it_), mapped_it_(other.mapped_it_) {}

  reference operator*() const noexcept { return ref(); }
  pointer operator->() const noexcept { return arrow_proxy(ref()); }

  reference operator[](difference_type n) const noexcept {
    return reference(*(key_it_ + n), *(mapped_it_ + n));
  }

  flat_map_iterator operator+(difference_type n) const noexcept {
    return flat_map_iterator(key_it_ + n, mapped_it_ + n);
  }
  flat_map_iterator operator-(difference_type n) const noexcept {
    return flat_map_iterator(key_it_ - n, mapped_it_ - n);
  }

  flat_map_iterator& operator++() noexcept {
    ++key_it_;
    ++mapped_it_;
    return *this;
  }
  flat_map_iterator operator++(int) noexcept {
    flat_map_iterator tmp(*this);
    ++key_it_;
    ++mapped_it_;
    return tmp;
  }

  flat_map_iterator& operator--() noexcept {
    --key_it_;
    --mapped_it_;
    return *this;
  }
  flat_map_iterator operator--(int) noexcept {
    flat_map_iterator tmp(*this);
    --key_it_;
    --mapped_it_;
    return tmp;
  }

  flat_map_iterator& operator+=(difference_type n) noexcept {
    key_it_ += n;
    mapped_it_ += n;
    return *this;
  }
  flat_map_iterator& operator-=(difference_type n) noexcept {
    key_it_ -= n;
    mapped_it_ -= n;
    return *this;
  }

  _KeyIter key_iter() const { return key_it_; }
  _MappedIter mapped_iter() const { return mapped_it_; }

  friend bool operator==(flat_map_iterator lhs, flat_map_iterator rhs) {
    return lhs.key_it_ == rhs.key_it_;
  }
  friend bool operator!=(flat_map_iterator lhs, flat_map_iterator rhs) {
    return !(lhs == rhs);
  }

  friend bool operator<(flat_map_iterator lhs, flat_map_iterator rhs) {
    return lhs.key_it_ < rhs.key_it_;
  }
  friend bool operator<=(flat_map_iterator lhs, flat_map_iterator rhs) {
    return lhs == rhs || lhs < rhs;
  }
  friend bool operator>(flat_map_iterator lhs, flat_map_iterator rhs) {
    return rhs < lhs;
  }
  friend bool operator>=(flat_map_iterator lhs, flat_map_iterator rhs) {
    return lhs == rhs || rhs < lhs;
  }

  friend typename flat_map_iterator::difference_type operator-(
      flat_map_iterator lhs, flat_map_iterator rhs) {
    return lhs.key_it_ - rhs.key_it_;
  }

 private:
  template <class _KeyRef2, class _TRef2, class _KeyIter2, class _MappedIter2>
  friend struct flat_map_iterator;

  reference ref() const { return reference(*key_it_, *mapped_it_); }

  _KeyIter key_it_;
  _MappedIter mapped_it_;
};

template <class _Key, class _T, class _Compare = std::less<_Key>,
          class _KeyContainer = std::vector<_Key>,
          class _MappedContainer = std::vector<_T>>
class flat_map {
 public:
  // types:
  using key_type = _Key;
  using mapped_type = _T;
  using value_type = std::pair<const key_type, mapped_type>;
  using key_compare = _Compare;
  using reference = std::pair<const key_type&, mapped_type&>;
  using const_reference = std::pair<const key_type&, const mapped_type&>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using iterator =
      flat_map_iterator<const key_type&, mapped_type&,
                        typename _KeyContainer::const_iterator,
                        typename _MappedContainer::iterator>;  // see 21.2
  using const_iterator =
      flat_map_iterator<const key_type&, const mapped_type&,
                        typename _KeyContainer::const_iterator,
                        typename _MappedContainer::const_iterator>;  // see 21.2
  using reverse_iterator = flat_map_iterator<
      const key_type&, mapped_type&,
      typename _KeyContainer::const_reverse_iterator,
      typename _MappedContainer::reverse_iterator>;  // see 21.2
  using const_reverse_iterator = flat_map_iterator<
      const key_type&, const mapped_type&,
      typename _KeyContainer::const_reverse_iterator,
      typename _MappedContainer::const_reverse_iterator>;  // see 21.2
  using key_container_type = _KeyContainer;
  using mapped_container_type = _MappedContainer;

  class value_compare {
    friend flat_map;

   private:
    key_compare comp;
    value_compare(key_compare c) : comp(c) {}  // NOLINT

   public:
    bool operator()(const_reference x, const_reference y) const {
      return comp(x.first, y.first);
    }
  };

  struct containers {
    key_container_type keys;
    mapped_container_type values;
  };

  // ??, construct/copy/destroy
  flat_map() {}

  // iterators
  iterator begin() noexcept {
    return iterator(c.keys.begin(), c.values.begin());
  }
  const_iterator begin() const noexcept {
    return const_iterator(c.keys.begin(), c.values.begin());
  }
  iterator end() noexcept { return iterator(c.keys.end(), c.values.end()); }
  const_iterator end() const noexcept {
    return const_iterator(c.keys.end(), c.values.end());
  }
  reverse_iterator rbegin() noexcept {
    return reverse_iterator(c.keys.rbegin(), c.values.rbegin());
  }
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(c.keys.rbegin(), c.values.rbegin());
  }
  reverse_iterator rend() noexcept {
    return reverse_iterator(c.keys.rend(), c.values.rend());
  }
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(c.keys.rend(), c.values.rend());
  }

  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }
  const_reverse_iterator crbegin() const noexcept { return rbegin(); }
  const_reverse_iterator crend() const noexcept { return rend(); }

  // ??, capacity
  [[nodiscard]]
  bool empty() const noexcept {
    return c.keys.empty();
  }
  size_type size() const noexcept { return c.keys.size(); }
  size_type max_size() const noexcept {
    return std::min<size_type>(c.keys.max_size(), c.values.max_size());
  }

  // ??, element access
  mapped_type& operator[](const key_type& x) {
    return try_emplace(x).first->second;
  }
  mapped_type& operator[](key_type&& x) {
    return try_emplace(std::move(x)).first->second;
  }

  // ??, modifiers

  template <class... _Args>
    requires std::is_constructible_v<mapped_type, _Args&&...>
  std::pair<iterator, bool> try_emplace(const key_type& k, _Args&&... args) {
    auto it = std::ranges::lower_bound(c.keys, k, compare);
    if (it == c.keys.end() || compare(*it, k) || compare(k, *it)) {
      auto values_it =
          c.values.emplace(project(it), std::forward<_Args>(args)...);
      it = c.keys.insert(it, k);
      return std::pair<iterator, bool>(iterator(it, values_it), true);
    }
    return std::pair<iterator, bool>(iterator(it, project(it)), false);
  }
  template <class... _Args>
    requires std::is_constructible_v<mapped_type, _Args&&...>
  std::pair<iterator, bool> try_emplace(key_type&& k, _Args&&... args) {
    auto it = std::ranges::lower_bound(c.keys, k, compare);
    if (it == c.keys.end() || compare(*it, k) || compare(k, *it)) {
      auto values_it =
          c.values.emplace(project(it), std::forward<_Args>(args)...);
      it = c.keys.insert(it, std::forward<key_type>(k));
      return std::pair<iterator, bool>(iterator(it, values_it), true);
    }
    return std::pair<iterator, bool>(iterator(it, project(it)), false);
  }
  template <class... _Args>
    requires std::is_constructible_v<mapped_type, _Args&&...>
  iterator try_emplace([[maybe_unused]] const_iterator hint, const key_type& k,
                       _Args&&... args) {
    return try_emplace(k, std::forward<_Args>(args)...).first;
  }
  template <class... _Args>
    requires std::is_constructible_v<mapped_type, _Args&&...>
  iterator try_emplace([[maybe_unused]] const_iterator hint, key_type&& k,
                       _Args&&... args) {
    return try_emplace(std::forward<key_type>(k), std::forward<_Args>(args)...)
        .first;
  }

  void clear() noexcept {
    c.keys.clear();
    c.values.clear();
  }

  // observers
  const key_container_type& keys() const noexcept { return c.keys; }
  const mapped_container_type& values() const noexcept { return c.values; }

  // map operations
  iterator find(const key_type& x) {
    auto it = key_find(x);
    return iterator(it, project(it));
  }
  const_iterator find(const key_type& x) const {
    auto it = key_find(x);
    return const_iterator(it, project(it));
  }
  template <class _K>
  iterator find(const _K& x) {
    auto it = key_find(x);
    return iterator(it, project(it));
  }
  template <class _K>
  const_iterator find(const _K& x) const {
    auto it = key_find(x);
    return iterator(it, project(it));
  }
  size_type count(const key_type& x) const {
    auto it = key_find(x);
    return size_type(it == c.keys.end() ? 0 : 1);
  }
  template <class _K>
  size_type count(const _K& x) const {
    auto it = key_find(x);
    return size_type(it == c.keys.end() ? 0 : 1);
  }
  bool contains(const key_type& x) const { return count(x) == size_type{1}; }
  template <class _K>
  bool contains(const _K& x) const {
    return count(x) == size_type{1};
  }

  friend bool operator==(const flat_map& x, const flat_map& y) {
    return std::ranges::equal(x, y);
  }
  friend bool operator!=(const flat_map& x, const flat_map& y) {
    return !(x == y);
  }

 private:
  containers c;         // exposition only
  key_compare compare;  // exposition only
  // exposition only
  struct scoped_clear {
    explicit scoped_clear(flat_map* fm) : fm_(fm) {}
    ~scoped_clear() {
      if (fm_) {
        fm_->clear();
      }
    }
    void release() { fm_ = nullptr; }

   private:
    flat_map* fm_;
  };

  using key_iter_t = typename _KeyContainer::iterator;
  using key_const_iter_t = typename _KeyContainer::const_iterator;
  using mapped_iter_t = typename _MappedContainer::iterator;
  using mapped_const_iter_t = typename _MappedContainer::const_iterator;

  using mutable_iterator =
      flat_map_iterator<key_type&, mapped_type&, key_iter_t, mapped_iter_t>;

  mapped_iter_t project(key_iter_t key_it) {
    return c.values.begin() + (key_it - c.keys.begin());
  }
  mapped_const_iter_t project(key_const_iter_t key_it) const {
    return c.values.begin() + (key_it - c.keys.begin());
  }

  template <typename _K>
  key_iter_t key_find(const _K& k) {
    auto it = std::ranges::lower_bound(c.keys, k, compare);
    if (it != c.keys.end() && (compare(*it, k) || compare(k, *it))) {
      it = c.keys.end();
    }
    return it;
  }
  template <typename _K>
  key_const_iter_t key_find(const _K& k) const {
    auto it = std::ranges::lower_bound(c.keys, k, compare);
    if (it != c.keys.end() && (compare(*it, k) || compare(k, *it))) {
      it = c.keys.end();
    }
    return it;
  }
};

}  // namespace slp
