# RCU Design for Autocache (Lock-Free Hit Path)

**Status**: Draft
**Date**: 2024-12-29
**Version**: 3.1
**Scope**: Make cache hits lock-free and thread-safe with copy-on-write snapshots using Julia's `@atomic` features.
**Prerequisites**: Julia 1.7+ (for `@atomic` field support). LTS (1.10) compatible.

## Goals

- **Lock-free Hit**: Cache hits (both Registry lookup and Bank lookup) must not acquire any locks.
- **Thread Safety**: Eliminate all race conditions on the hit path.
- **Zero Overhead (Steady State)**: No allocations or heavy operations once the cache is warm.
- **Preserve Semantics**: Maintain existing LRU-like (Ring Buffer) behavior.

## Non-Goals

- **Lock-free Miss**: Cache misses (writes) will use a lock.
- **Zero Allocation on Miss**: Misses will involve small allocations (snapshot copy), which is acceptable as misses are rare.

## Core Concept: Read-Copy-Update (RCU)

The design uses an **Immutable Snapshot** pattern for both the **Registry** and the **CacheBanks**.
- **Readers**: Atomically load a reference to the current snapshot. Since the snapshot is immutable, they can read it safely without locks.
- **Writers**: Acquire a lock, copy the current snapshot, modify the copy, and atomically publish the new snapshot.

## Data Structures

### 1. Bank Snapshot
Holds the state of a single cache bank at a specific point in time.

```julia
struct BankSnapshot{E}
    store::Vector{E}  # Immutable content (effectively)
    count::Int        # Number of valid entries
    ring::Int         # Next eviction index (1-based)
end
```

### 2. Registry Snapshot
Holds the mapping of types to banks. Replaces `IdDict` to ensure safe concurrent reads.
Since the number of banks is small (< 20), a simple `Vector` is efficient.

```julia
# A simple association list: [(BankType, BankInstance), ...]
const RegistrySnapshot = Vector{Pair{DataType, Any}}
```

### 3. Atomic Structures
Holds the atomic reference to the current snapshots.

```julia
# 1. The Registry (Global)
# Wrapper struct to support @atomic field for reference types (Vector)
mutable struct GlobalRegistry
    @atomic snapshot::RegistrySnapshot
end

# Replaces: const _DERIVATIVE_BANK_REGISTRY = IdDict{DataType, Any}()
const _DERIVATIVE_REGISTRY = GlobalRegistry(RegistrySnapshot())

# 2. The Bank (Per-Type)
mutable struct CacheBank{E<:AbstractCacheEntry}
    @atomic snapshot::BankSnapshot{E} # Atomic reference
end
```

## Implementation Details

### 1. Registry Lookup (RCU)

Replaces the unsafe `IdDict` + `sizehint!` approach.

```julia
@inline function _get_bank_rcu(registry::GlobalRegistry, BankType::DataType)
    # 1. Atomic Load (Acquire)
    # Explicit memory ordering ensures we see a fully initialized snapshot
    snapshot = @atomic :acquire registry.snapshot
    
    # 2. Linear Scan (Fast for small N < 20)
    # If N grows significantly, consider sorted vector + binary search
    for (TypeKey, Bank) in snapshot
        if TypeKey === BankType
            return Bank
        end
    end
    return nothing
end

# Write Path (Creation)
function _get_or_create_bank!(registry::GlobalRegistry, BankType)
    # Fast Path
    bank = _get_bank_rcu(registry, BankType)
    bank !== nothing && return bank

    lock(_CACHE_LOCK)
    try
        # Double Check
        bank = _get_bank_rcu(registry, BankType)
        bank !== nothing && return bank

        # Create new bank
        new_bank = CacheBank{...}()
        
        # RCU Update
        old_snap = @atomic :monotonic registry.snapshot # Lock held, monotonic ok
        new_snap = copy(old_snap)
        push!(new_snap, BankType => new_bank)
        
        # Atomic Publish (Release)
        @atomic :release registry.snapshot = new_snap
        
        return new_bank
    finally
        unlock(_CACHE_LOCK)
    end
end

# Clearing the Cache
function clear_cubic_cache!()
    lock(_CACHE_LOCK)
    try
        # Atomically replace with empty snapshot
        # Old snapshots held by running tasks remain valid until they finish
        @atomic :release _DERIVATIVE_REGISTRY.snapshot = RegistrySnapshot()
    finally
        unlock(_CACHE_LOCK)
    end
end
```

### 2. Bank Lookup (RCU)

```julia
@inline function _lookup(bank::CacheBank{E}, id::UInt, x::X) where {E,X}
    # 1. Atomic Load (Acquire Ordering)
    snap = @atomic :acquire bank.snapshot
    store = snap.store
    count = snap.count

    # 2. Pass 1: Identity Check
    @inbounds for i in 1:count
        entry = store[i]
        if entry.id === id
            return entry.cache
        end
    end

    # 3. Pass 2: Equality Check
    @inbounds for i in 1:count
        entry = store[i]
        if isequal(entry.x, x)
            # RULE: Strict Immutability in MT
            # Self-healing is ONLY allowed in single-threaded mode.
            # Modifying entry.id in MT would race with other readers.
            if Threads.nthreads() == 1
                entry.id = id
            end
            return entry.cache
        end
    end

    return nothing
end
```

### 3. Bank Insert (RCU)

```julia
@inline function _lookup_or_insert!(bank::CacheBank{E}, x::X, bc_data) where {E,X}
    # Fast Path (Lock-free)
    found = _lookup(bank, objectid(x), x)
    found !== nothing && return found

    lock(_CACHE_LOCK)
    try
        # Double-Check
        found = _lookup(bank, objectid(x), x)
        found !== nothing && return found

        # Build new cache (Expensive operation)
        new_cache = _build_cache(E, x, bc_data)
        new_entry = E(objectid(x), x, new_cache)

        # --- RCU Update ---
        # 1. Load current snapshot (Monotonic is enough inside lock)
        old_snap = @atomic :monotonic bank.snapshot
        
        # 2. Copy (O(N), N=16 -> Negligible cost)
        new_store = copy(old_snap.store)
        new_count = old_snap.count
        new_ring = old_snap.ring

        # 3. Modify Copy
        if new_count < _CACHE_SIZE
            push!(new_store, new_entry)
            new_count += 1
        else
            new_store[new_ring] = new_entry
            new_ring = (new_ring % _CACHE_SIZE) + 1
        end

        # 4. Publish (Release Ordering)
        new_snap = BankSnapshot(new_store, new_count, new_ring)
        @atomic :release bank.snapshot = new_snap
        # ------------------

        return new_cache
    finally
        unlock(_CACHE_LOCK)
    end
end
```

## Memory Ordering & Safety

- **Acquire/Release**: Ensures that when a reader sees the new `snapshot` pointer, all writes to `new_store` (and the `new_entry` inside it) are visible.
- **GC Safety**: Old snapshots are naturally garbage collected when no readers hold a reference to them.
- **Atomic Field**: Julia 1.7+ `@atomic` fields handle the necessary memory barriers and object reference atomicity.

## Julia Version Compatibility

### Atomic Fields vs Atomic Reference (Important Clarification)

Julia 문서에는 다음과 같이 명시되어 있음:
- "Atomic fields functionality requires at least Julia 1.7"
- "Atomic reference functionality requires at least Julia 1.12"

**이 두 가지는 다른 기능을 의미함:**

| 기능 | 문법 | 필요 버전 | 설명 |
|------|------|----------|------|
| **Atomic fields** | `@atomic a.field` | 1.7+ | struct 필드에 대한 atomic 접근 |
| **Atomic reference** | `@atomic mem[idx]` | 1.12+ | `AtomicMemory{T}` 인덱스 접근 |

### 우리 디자인의 호환성

```julia
# 우리가 사용하는 패턴 (Atomic field - Julia 1.7+)
mutable struct CacheBank{E}
    @atomic snapshot::BankSnapshot{E}
end

snap = @atomic :acquire bank.snapshot  # atomic field load
@atomic :release bank.snapshot = new_snap  # atomic field store
```

이것은 **atomic field access**이며, non-isbits 타입(`BankSnapshot`)을 포함해도 Julia 1.7+에서 안전하게 작동함.

### 검증 결과 (2025-12-29)

다음 조건에서 **torn read 테스트 통과**:
- Julia 1.10 (LTS): 154,682 reads, 38,671 writes, **0 torn reads** ✅
- Julia 1.12: 146,915 reads, 36,729 writes, **0 torn reads** ✅

테스트 방법:
- 4 threads, 3초간 continuous read/write
- Writer: 매 write마다 새 BankSnapshot 생성 (batch ID 포함)
- Reader: snapshot 내 모든 값이 동일 batch인지 검증

## Performance Considerations

- **Hit**: 
    - Registry: 1 Atomic Load + Linear Scan (N < 20).
    - Bank: 1 Atomic Load + Linear Scan (N < 16).
    - Total: Extremely fast, zero lock contention.
- **Miss**: 
    - 1 Global Lock + Vector Copy + Atomic Store.
    - **Trade-off**: Global lock serializes misses across all banks. This is acceptable as misses converge quickly.

## Migration Plan (Phase 3)

1.  **Registry**: Replace `IdDict` with `Atomic{Vector{Pair{DataType, Any}}}`.
2.  **Bank**: Define `BankSnapshot` and update `CacheBank` to use `@atomic snapshot`.
3.  **Logic**: Implement RCU lookup/insert logic.
4.  **Verify**: Run `test/test_thread_safety.jl`.
