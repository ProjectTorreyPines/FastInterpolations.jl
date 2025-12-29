# Lock-Free Hit Designs for Autocache (Option B vs C)

Status: Draft
Date: 2024-12-29
Scope: Autocache hit path safety in multithreaded use

## Goals

- Keep cache hit path lock-free and thread-safe.
- Avoid shared mutable state in the hit path.
- Preserve current API semantics and cache behavior where possible.

## Non-Goals

- Lock-free cache miss (miss can be slower).
- Perfect cache reuse across all threads with zero memory cost.

## Context

The current cache uses:
- A registry (`IdDict`) from BankType to Bank.
- Banks contain a `store::Vector{CacheEntry}` and a `ring` index.
- Hits are found by scanning `store`.

Lock-free hit is only safe if the data read by the hit path is never mutated
while any thread is reading it. That means either:
- no writes at all during multithreaded use, or
- writes only happen by publishing new immutable snapshots.

Below are two concrete designs that satisfy this.

---

## Option B: Per-Thread or Per-Task Cache (TLS)

### Summary

Each thread or task owns its own registry and banks. There is no sharing across
threads/tasks. Hits and misses are lock-free because no other thread touches
the same data.

### Data Structures

- `registry`: per-thread or per-task `IdDict{DataType, Any}`
- `bank`: per-thread or per-task `CacheBank` with `store` and `ring`
- no global `_DERIVATIVE_BANK_REGISTRY` in multithreaded mode

### Variants

1. Per-thread registry (simpler, fewer registries)
   - Use `Vector{IdDict}` indexed by `Threads.threadid()`.
   - Safe because each thread only uses its own registry.
   - Task migration between threads reduces cache reuse, but is correct.

2. Per-task registry (strict isolation)
   - Use `task_local_storage()` with a fixed key.
   - Strongest correctness guarantee, but more registries and memory use.

### Pseudocode (Per-thread)

```julia
const _THREAD_REGISTRY = [IdDict{DataType, Any}() for _ in 1:Threads.nthreads()]

@inline function _thread_registry()
    return _THREAD_REGISTRY[Threads.threadid()]
end

@inline function _get_derivative_bank(x, bc_pair)
    reg = _thread_registry()
    BankType = CacheBank{T,L,R,X}
    bank = get(reg, BankType, nothing)
    bank === nothing && (reg[BankType] = CacheBank{T,L,R,X}())
    return reg[BankType]::BankType
end
```

### Thread Safety Argument

Only one thread ever reads or writes a given registry/bank. Therefore:
- hits are lock-free and safe
- misses are lock-free and safe
- no data races in `store` or `ring`

### Performance and Trade-offs

Pros:
- Easiest safe lock-free hit
- No locking in hit or miss
- Very low implementation risk

Cons:
- Duplicated caches across threads/tasks
- Higher memory cost if `x` is large and reused across threads
- Cache reuse across threads is lost

When it is good:
- Many single-threaded users
- Multithreaded users do not share `x` across threads
- Memory is not a primary constraint

---

## Option C: Copy-On-Write Snapshot (RCU-style)

### Summary

Hits read from an immutable snapshot. Misses create a new snapshot and
atomically publish it. Readers never see partially updated data.

### Data Structures

```julia
struct BankSnapshot{E}
    store::Vector{E}   # immutable after creation
    ring::Int          # immutable after creation
end

mutable struct CacheBank{E}
    snapshot::Atomic{BankSnapshot{E}}
end
```

Notes:
- `snapshot` must be updated atomically.
- Snapshot objects are immutable after construction.

### Read Path (Hit)

```julia
snap = atomic_load(bank.snapshot)  # acquire
for entry in snap.store
    entry.id === id && return entry.cache
end
for entry in snap.store
    isequal(entry.x, x) && return entry.cache
end
```

### Write Path (Miss)

```julia
old = atomic_load(bank.snapshot)
new_store = copy(old.store)
new_ring = old.ring
insert!(new_store, new_ring, new_entry)  # ring update
new = BankSnapshot(new_store, new_ring)
atomic_store!(bank.snapshot, new)        # release
return new_entry.cache
```

### Thread Safety Argument

- Readers only access immutable snapshots.
- Writers publish new snapshots atomically.
- No concurrent mutation of any snapshot seen by readers.

### Performance and Trade-offs

Pros:
- Lock-free hit and safe
- Cache is shared across threads
- Memory footprint converges to single snapshot per bank

Cons:
- Miss does an O(capacity) copy (store size), even if capacity is 16
- Two threads can compute the same cache concurrently (wasted work)
- Requires atomic field support and careful memory ordering

When it is good:
- Multithreaded users share the same `x` across threads
- Memory cost of per-thread caches is too high
- You can accept slightly higher miss cost for safe lock-free hits

---

## Atomic Support Note (Option C)

Publishing a new snapshot must be done with atomics. This requires:
- `Atomic{T}` support for the snapshot type, or
- `@atomic` field access (Julia 1.9+), or
- a `Ptr`-based atomic with GC-safe handling (higher risk)

If atomic snapshot storage is not available, Option C is not safe.

---

## Trade-off Summary

| Criterion | Option B (TLS) | Option C (Snapshot) |
|---|---|---|
| Hit path | Lock-free, safe | Lock-free, safe |
| Miss cost | Low | O(capacity) copy |
| Cache sharing | None | Full |
| Memory use | High (per thread/task) | Low (shared) |
| Implementation risk | Low | Medium to high |
| Atomic complexity | None | Required |

---

## Recommendation

Short term: Option B (per-thread registry) is the safest and simplest way to
get lock-free hits with low implementation risk.

Medium term: Option C is attractive if multithreaded users commonly share
large `x` grids and memory duplication becomes a problem. It is more complex
and requires solid atomic support in the target Julia version.

Suggested path:
1) Implement Option B to unblock correctness and performance quickly.
2) Reassess Option C if memory duplication becomes a real user issue.
