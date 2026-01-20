# Changelog

All notable changes to FastInterpolations.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Refactored interval search to typed `SearchPolicy` system for compile-time dispatch
- Anchor batch operations (`_fill_anchors!`) now use `LinearBounded{8}` for improved monotonic query performance

### Internal
- Added `src/core/search.jl` with `Searcher{P,H}` type system
- Search policies: `Binary`, `HintedBinary`, `LinearBounded{N}`
- Hint types: `NoHint`, `RefHint`
- Zero-overhead: default path compiles to identical assembly as before
- Renamed `_search_binary(::AbstractRange)` → `_search_direct` for clarity (O(1) direct calculation, not binary search)

### Removed
- `_find_interval(x, xi)` — replaced by `_search_interval(x, xi)`
- `_find_interval(x, spacing, xi)` — replaced by `_search_interval(x, spacing, xi)`
