# [Memory & Allocation](@id memory_allocation)

Optimizing memory usage in FastInterpolations.jl focuses on three key areas: vector handling, allocation patterns, and grid selection.

## 1. Vector Queries: Avoid Loops

When querying multiple points (`xq` is a vector), **never loop over scalar queries**.

**Bad:**
```julia
results = zeros(length(xq))
for i in eachindex(xq)
   results[i] = linear_interp(x, y, xq[i]) # High overhead per call
end
```

**Good (Vector API):**
```julia
results = linear_interp(x, y, xq) # Optimized batch processing
```

The vector API handles grid search and overhead much more efficiently internally.

## 2. Zero-Allocation: Pre-allocate & In-place

To achieve **true zero-allocation**, combine the Vector API with pre-allocated output arrays using the `!` (bang) versions of the functions.

**Good (Standard Vector API):**
```julia
# Allocates a new array for 'results' every time
results = linear_interp(x, y, xq) 
```

**Best (Zero-Allocation):**
```julia
# 1. Pre-allocate output array (once, outside loop)
out = similar(xq)

# 2. Use in-place version inside loop
linear_interp!(out, x, y, xq)
```

Use this pattern in hot loops to eliminate garbage collection overhead.

## 3. Grid Selection: Range vs Vector

For uniform grids, always use `Range` instead of `Vector`:

```julia
# ✅ Good: Range (O(1) lookup, minimal memory)
x = 0.0:0.1:10.0

# ❌ Bad: collect() degrades to Vector (O(log n) lookup)
x = collect(0.0:0.1:10.0)
```

For non-uniform grids where `Vector` is required, see [Search & Hints](@ref search_hints) for optimizing lookup performance.
