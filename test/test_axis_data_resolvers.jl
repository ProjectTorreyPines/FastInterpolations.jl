@testitem "_resolve_axis — type correctness across (grid, bc) combinations" begin
    using FastInterpolations:
        _resolve_axis, _resolve_data,
        _CachedRange, _CachedVector, _ExclusivePeriodicAxis, _ExclusivePeriodicData,
        NoBC, PeriodicBC

    bc_no = NoBC()
    bc_inc = PeriodicBC(endpoint = :inclusive)
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "Vector + non-exclusive → passthrough" begin
        x = [0.0, 1.0, 2.0, 3.0]
        @test _resolve_axis(x, bc_no) === x          # zero-alloc passthrough
        @test _resolve_axis(x, bc_inc) === x          # zero-alloc passthrough
    end

    @testset "Vector + :exclusive → _ExclusivePeriodicAxis" begin
        x = [0.0, 1.0, 2.0, 3.0]
        ax = _resolve_axis(x, bc_excl)
        @test ax isa _ExclusivePeriodicAxis{Float64, Vector{Float64}, Float64}
        @test ax.inner === x                         # zero-copy reference
        @test ax.period == 4.0
        @test length(ax) == 5                        # virtual n+1
    end

    @testset "Range + non-exclusive → _CachedRange (length n)" begin
        r = 0.0:1.0:3.0
        cr = _resolve_axis(r, bc_no)
        @test cr isa _CachedRange{Float64}
        @test length(cr) == 4

        cr_inc = _resolve_axis(r, bc_inc)
        @test cr_inc isa _CachedRange{Float64}
        @test length(cr_inc) == 4
    end

    @testset "Range + :exclusive → _ExclusivePeriodicAxis(_CachedRange, period)" begin
        r = 0.0:1.0:3.0
        cr = _resolve_axis(r, bc_excl)
        @test cr isa FastInterpolations._ExclusivePeriodicAxis
        @test cr.inner isa _CachedRange{Float64}
        @test length(cr) == 5                        # virtual length n+1
        @test length(cr.inner) == 4                  # raw n-length cached Range
        @test last(cr) ≈ 4.0                         # = inner[1] + period
    end
end

@testitem "_resolve_axis 3-arg Tg dispatch table (incl. pre-wrapped × :exclusive diagonal)" begin
    using FastInterpolations:
        _resolve_axis, _CachedRange, _CachedVector, _ExclusivePeriodicAxis,
        _to_float, NoBC, PeriodicBC

    # 3-arg `_resolve_axis(x, bc, Tg)` dispatch table. The pre-wrapped × `:exclusive`
    # DIAGONAL methods are load-bearing against ambiguity: `_Cached* <: Abstract*`, so
    # (specific container × generic BC) and (generic container × specific BC) cross
    # without them — Aqua ambiguity RED + runtime MethodError in hetero one-shot.
    bc_no = NoBC()
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "raw Range + Tg → _CachedRange{Tg}" begin
        @test _resolve_axis(0:1:3, bc_no, Float32) isa _CachedRange{Float32}
        @test _resolve_axis(0.0:1.0:3.0, bc_no, Float64) isa _CachedRange{Float64}
    end

    @testset "_CachedRange + non-exclusive + Tg → _convert_copy (same-type identity)" begin
        cr64 = _to_float(0.0:1.0:3.0, Float64)
        @test _resolve_axis(cr64, bc_no, Float64) === cr64
        @test _resolve_axis(cr64, bc_no, Float32) isa _CachedRange{Float32}
    end

    @testset "raw Vector / _CachedVector + non-exclusive + Tg → passthrough (Tg ignored)" begin
        x32 = Float32[0.0, 1.0, 2.0, 3.0]
        cv64 = _CachedVector([0.0, 1.0, 2.0, 3.0])
        @test _resolve_axis(x32, bc_no, Float64) === x32
        @test _resolve_axis(cv64, bc_no, Float32) === cv64
    end

    @testset "raw Range/Vector + :exclusive + Tg → _ExclusivePeriodicAxis" begin
        ax_r = _resolve_axis(0:1:3, bc_excl, Float32)
        @test ax_r isa _ExclusivePeriodicAxis
        @test ax_r.inner isa _CachedRange{Float32}    # Tg respected on raw Range
        # Period follows the Tg-typed axis (`_resolve_bc_period` normalizes it): a
        # Float64 period literal must NOT re-widen a value-matched Float32 axis.
        @test ax_r.period isa Float32
        @test eltype(ax_r) === Float32
        x = [0.0, 1.0, 2.0, 3.0]
        ax_v = _resolve_axis(x, bc_excl, Float64)
        @test ax_v isa _ExclusivePeriodicAxis
        @test ax_v.inner === x                        # raw Vector never converts
    end

    @testset "DIAGONAL: _CachedRange + :exclusive + Tg" begin
        cr64 = _to_float(0.0:1.0:3.0, Float64)
        ax = _resolve_axis(cr64, bc_excl, Float64)
        @test ax isa _ExclusivePeriodicAxis
        @test ax.inner === cr64                       # same-type `_convert_copy` identity
        @test ax.period == 4.0
        @test length(ax) == 5                         # virtual n+1
        # Cross-eltype: the axis converts to Tg first, and the period is resolved
        # against the CONVERTED axis — so it follows Tg instead of re-widening.
        ax32 = _resolve_axis(cr64, bc_excl, Float32)
        @test ax32.inner isa _CachedRange{Float32}
        @test ax32.period isa Float32
    end

    @testset "DIAGONAL: _CachedVector + :exclusive + Tg" begin
        cv64 = _CachedVector([0.0, 1.0, 2.0, 3.0])
        ax = _resolve_axis(cv64, bc_excl, Float64)
        @test ax isa _ExclusivePeriodicAxis
        @test ax.inner isa _CachedVector{Float64}     # Tg respected (same-width converted copy)
        @test ax.period == 4.0
        # `:exclusive` converts to Tg — UNLIKE a non-periodic vector, the wrapped axis eltype flows
        # into the value/witness path via the period seam, so it must follow Tg (matching the
        # `_CachedRange` diagonal above and the raw-Vector arm). The old "never convert" contract
        # produced a Float64 axis past a value-matched Float32 witness → crash.
        ax32 = _resolve_axis(cv64, bc_excl, Float32)
        @test ax32.inner isa _CachedVector{Float32}
        @test ax32.period isa Float32
    end

    @testset "diagonals are type-stable" begin
        cr64 = _to_float(0.0:1.0:3.0, Float64)
        cv64 = _CachedVector([0.0, 1.0, 2.0, 3.0])
        f(x, bc, ::Type{T}) where {T} = _resolve_axis(x, bc, T)
        @test (@inferred f(cr64, bc_excl, Float64)) isa _ExclusivePeriodicAxis
        @test (@inferred f(cv64, bc_excl, Float64)) isa _ExclusivePeriodicAxis
    end

    @testset "Tg-typed 2-arg (no BC): Ranges value-match, Vectors pass through" begin
        # Used by the 1D one-shot entries (no bc concept at the normalize point).
        @test _resolve_axis(0:1:3, Float32) isa _CachedRange{Float32}
        cr64 = _to_float(0.0:1.0:3.0, Float64)
        @test _resolve_axis(cr64, Float64) === cr64       # same-type identity
        @test _resolve_axis(cr64, Float32) isa _CachedRange{Float32}
        x32 = Float32[0.0, 1.0, 2.0, 3.0]
        @test _resolve_axis(x32, Float64) === x32         # vectors never convert
        cv64 = _CachedVector([0.0, 1.0, 2.0, 3.0])
        @test _resolve_axis(cv64, Float32) === cv64
    end
end

@testitem "_resolve_data — type correctness" begin
    using FastInterpolations: _resolve_data, _ExclusivePeriodicData, NoBC, PeriodicBC

    y = [10.0, 20.0, 30.0, 40.0]
    bc_no = NoBC()
    bc_inc = PeriodicBC(endpoint = :inclusive, check = false)
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "non-exclusive → passthrough" begin
        @test _resolve_data(y, bc_no) === y
        @test _resolve_data(y, bc_inc) === y         # check=false skips validation
    end

    @testset ":exclusive → _ExclusivePeriodicData (zero-copy ref)" begin
        yd = _resolve_data(y, bc_excl)
        @test yd isa _ExclusivePeriodicData{Float64, 1, Vector{Float64}}
        @test yd.inner === y                         # zero-copy reference
        @test length(yd) == 5                        # virtual n+1
    end

    @testset ":inclusive endpoint validation when check=true" begin
        # `check` is a type-parameter on `PeriodicBC{E,P,C}` — read via
        # `periodic_check(bc)` (zero-cost). `check=true` is the default.
        bc_inc_check = PeriodicBC(endpoint = :inclusive)   # check=true (default)
        # Bad endpoint raises
        y_bad = [10.0, 20.0, 30.0, 99.0]                    # last ≠ first
        @test_throws ArgumentError _resolve_data(y_bad, bc_inc_check)
        # Good endpoint passes
        y_good = [10.0, 20.0, 30.0, 10.0]
        @test _resolve_data(y_good, bc_inc_check) === y_good
    end
end

@testitem "_cache_axis(_convert_copy(x, T), bc) — persistent build pattern (cubic-style)" begin
    using FastInterpolations:
        _cache_axis, _convert_copy, _CachedRange, _CachedVector, _ExclusivePeriodicAxis,
        NoBC, PeriodicBC

    # Canonical owned-axis pattern used by `_build_derivative_bc_cache` /
    # `_build_periodic_cache` in cubic_solver.jl: copy buffer + promote eltype
    # (one allocation pass), then alias the fresh buffer and build h/inv_h.

    bc_no = NoBC()
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "Vector + non-exclusive → _CachedVector (owned)" begin
        x = [0.0, 1.0, 2.0, 3.0]
        xc = _cache_axis(_convert_copy(x, Float64), bc_no)
        @test xc isa _CachedVector{Float64, Float64}
        @test xc.inner == x                          # copied
        @test xc.inner !== x                         # NOT same object (mutation-safe)
        @test xc.h ≈ [1.0, 1.0, 1.0]                 # cached spacing
    end

    @testset "Vector + :exclusive → _ExclusivePeriodicAxis{_CachedVector inner}" begin
        x = [0.0, 1.0, 2.0, 3.0]
        ax = _cache_axis(_convert_copy(x, Float64), bc_excl)
        @test ax isa _ExclusivePeriodicAxis{Float64, _CachedVector{Float64, Float64}, Float64}
        @test ax.inner isa _CachedVector
        @test ax.inner.h ≈ [1.0, 1.0, 1.0]           # inner is cached
        @test ax.period == 4.0
        @test length(ax) == 5                         # virtual n+1
    end

    @testset "Range + non-exclusive → _CachedRange (length n)" begin
        r = 0.0:1.0:3.0
        cr = _cache_axis(_convert_copy(r, Float64), bc_no)
        @test cr isa _CachedRange{Float64}
        @test length(cr) == 4
    end

    @testset "Range + :exclusive → _ExclusivePeriodicAxis(_CachedRange, period)" begin
        r = 0.0:1.0:3.0
        cr = _cache_axis(_convert_copy(r, Float64), bc_excl)
        @test cr isa FastInterpolations._ExclusivePeriodicAxis
        @test cr.inner isa _CachedRange{Float64}
        @test length(cr) == 5
        @test length(cr.inner) == 4
    end

    @testset "Float32 → Float64 eltype promotion" begin
        # Vector path
        x32 = Float32[1.0, 2.0, 3.0]
        out = _cache_axis(_convert_copy(x32, Float64), bc_no)
        @test out isa _CachedVector{Float64, Float64}
        @test eltype(out) === Float64

        # _CachedRange path — cubic flow pre-normalizes Range via outer
        # `_resolve_axis(x)` before reaching this pattern, so the input here
        # is `_CachedRange`, not a raw `AbstractRange`. The same-shape rebuild
        # preserves Range type through `_convert_copy(::_CachedRange, T)`.
        cr32 = FastInterpolations._to_float(0.0f0:1.0f0:3.0f0, Float32)
        out_r = _cache_axis(_convert_copy(cr32, Float64), bc_no)
        @test out_r isa _CachedRange{Float64}
    end
end

@testitem "Resolvers are type-stable when used inside a function" begin
    using FastInterpolations:
        _resolve_axis, _resolve_data, _cache_axis, _convert_copy,
        _CachedRange, _CachedVector, _ExclusivePeriodicAxis, _ExclusivePeriodicData,
        NoBC, PeriodicBC

    # Each test function takes concrete-typed inputs. We assert the return type
    # via `@inferred` — Julia's compiler must determine the concrete output type
    # purely from input types (no runtime dispatch on `bc`/`x`).

    @testset "_resolve_axis type stability" begin
        x_vec = [0.0, 1.0, 2.0, 3.0]
        x_rng = 0.0:1.0:3.0
        bc_no = NoBC()
        bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

        f_vec_nobc(x, bc) = _resolve_axis(x, bc)
        @test (@inferred f_vec_nobc(x_vec, bc_no)) === x_vec

        f_rng_nobc(x, bc) = _resolve_axis(x, bc)
        @inferred f_rng_nobc(x_rng, bc_no)            # _CachedRange
        @test f_rng_nobc(x_rng, bc_no) isa _CachedRange{Float64}

        f_vec_excl(x, bc) = _resolve_axis(x, bc)
        @inferred f_vec_excl(x_vec, bc_excl)          # _ExclusivePeriodicAxis
        @test f_vec_excl(x_vec, bc_excl) isa _ExclusivePeriodicAxis{Float64}

        f_rng_excl(x, bc) = _resolve_axis(x, bc)
        @inferred f_rng_excl(x_rng, bc_excl)          # _CachedRange of length n+1
        @test length(f_rng_excl(x_rng, bc_excl)) == 5
    end

    @testset "_resolve_data type stability" begin
        y = [10.0, 20.0, 30.0, 40.0]
        bc_no = NoBC()
        bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

        f(y, bc) = _resolve_data(y, bc)
        @test (@inferred f(y, bc_no)) === y
        @inferred f(y, bc_excl)
        @test f(y, bc_excl) isa _ExclusivePeriodicData{Float64, 1, Vector{Float64}}
    end

    @testset "_cache_axis(_convert_copy(x, Tg), bc) type stability (cubic build pattern)" begin
        x_vec = [0.0, 1.0, 2.0, 3.0]
        x_rng = 0.0:1.0:3.0
        bc_no = NoBC()
        bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

        f(x, bc, Tg) = _cache_axis(_convert_copy(x, Tg), bc)
        @inferred f(x_vec, bc_no, Float64)
        @inferred f(x_vec, bc_excl, Float64)
        @inferred f(x_rng, bc_no, Float64)
        @inferred f(x_rng, bc_excl, Float64)
        @test f(x_vec, bc_no, Float64) isa _CachedVector{Float64, Float64}
        @test f(x_vec, bc_excl, Float64) isa
            _ExclusivePeriodicAxis{Float64, _CachedVector{Float64, Float64}}
    end
end

@testitem "_cache_axis — persistent surface API contract" begin
    using FastInterpolations:
        _cache_axis,
        _CachedRange, _CachedVector, _ExclusivePeriodicAxis,
        NoBC, PeriodicBC

    bc_no = NoBC()
    bc_inc = PeriodicBC(endpoint = :inclusive)
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "Vector + non-exclusive → _CachedVector aliasing inner" begin
        # Persistent surface API contract: zero-copy of user's data buffer.
        # The wrapper allocates fresh `h`/`inv_h` but `inner` is the SAME
        # `Vector{Float64}` object the user passed in. Mutation safety is
        # the inner constructor's responsibility (`_convert_copy`).
        x = [0.0, 1.0, 2.0, 3.0]
        c_no = _cache_axis(x, bc_no)
        c_inc = _cache_axis(x, bc_inc)
        @test c_no isa _CachedVector{Float64, Float64}
        @test c_inc isa _CachedVector{Float64, Float64}
        @test c_no.inner === x          # ALIAS, not copy
        @test c_inc.inner === x         # ALIAS, not copy
        @test c_no.h ≈ [1.0, 1.0, 1.0]  # cached spacing
        @test c_no.inv_h ≈ [1.0, 1.0, 1.0]
    end

    @testset "Vector + :exclusive → _ExclusivePeriodicAxis(_CachedVector aliasing)" begin
        x = [0.0, 1.0, 2.0, 3.0]
        ax = _cache_axis(x, bc_excl)
        @test ax isa _ExclusivePeriodicAxis
        @test ax.inner isa _CachedVector{Float64, Float64}
        @test ax.inner.inner === x      # double-aliasing through to user
        @test ax.period == 4.0
        @test length(ax) == 5            # virtual n+1
    end

    @testset "Range + non-exclusive → _CachedRange (immutable, zero-buffer)" begin
        r = 0.0:1.0:3.0
        cr = _cache_axis(r, bc_no)
        @test cr isa _CachedRange{Float64}
        @test length(cr) == 4
    end

    @testset "Range + :exclusive → _ExclusivePeriodicAxis(_CachedRange, period)" begin
        r = 0.0:1.0:3.0
        ax = _cache_axis(r, bc_excl)
        @test ax isa _ExclusivePeriodicAxis
        @test ax.inner isa _CachedRange{Float64}
        @test length(ax) == 5
    end

    @testset "Pre-wrapped re-entry: idempotent passthrough (===)" begin
        # `_cache_axis` on already-wrapped inputs returns the wrapper as-is.
        # Mutation safety in this case relies on the downstream inner ctor's
        # `_convert_copy(c, Tg) = Base.copy(c)` taking ownership.
        x_cv = _CachedVector([0.0, 1.0, 2.0])
        @test _cache_axis(x_cv, bc_no) === x_cv
        @test _cache_axis(x_cv, bc_inc) === x_cv

        x_cr = FastInterpolations._to_float(0.0:1.0:3.0, Float64)
        @test _cache_axis(x_cr, bc_no) === x_cr
        @test _cache_axis(x_cr, bc_inc) === x_cr

        x_ax = _ExclusivePeriodicAxis([0.0, 1.0, 2.0], 3.0)
        @test _cache_axis(x_ax, bc_no) === x_ax
        @test _cache_axis(x_ax, bc_excl) === x_ax
    end

    @testset "Pre-wrapped + :exclusive: re-wrap into _ExclusivePeriodicAxis" begin
        # _CachedVector + :exclusive should produce _ExclusivePeriodicAxis
        # wrapping that cached vector (still aliasing — inner.inner unchanged).
        x_cv = _CachedVector([0.0, 1.0, 2.0, 3.0])
        ax = _cache_axis(x_cv, bc_excl)
        @test ax isa _ExclusivePeriodicAxis
        @test ax.inner === x_cv         # cached vector reused as-is
        @test ax.period == 4.0
    end
end

@testitem "_cache_axis is type-stable (no Union returns)" begin
    using FastInterpolations: _cache_axis, _CachedRange, _CachedVector, NoBC, PeriodicBC

    # Regression guard: `_cache_axis` must not return a Union type for any
    # input. (Earlier `length(x) >= 2 ? _CachedVector(x) : x` branch caused
    # `Union{_CachedVector, Vector}` inference; fix moved singleton handling
    # to the HeteroND-only `_cache_axis_for_method` helper.)
    bc_no = NoBC()
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    f(x, bc) = _cache_axis(x, bc)
    @test (@inferred f([0.0, 1.0, 2.0, 3.0], bc_no)) isa _CachedVector{Float64, Float64}
    @test (@inferred f(0.0:1.0:3.0, bc_no)) isa _CachedRange{Float64}
    @inferred f([0.0, 1.0, 2.0, 3.0], bc_excl)
    @inferred f(0.0:1.0:3.0, bc_excl)
end

@testitem "Resolvers are zero-alloc on hot path (oneshot use)" setup = [AllocConstants] begin
    using FastInterpolations: _resolve_axis, _resolve_data, NoBC, PeriodicBC

    # ALLOC_THRESHOLD comes from `AllocConstants`: 0 on Julia ≥ 1.12, ~240 B on
    # LTS to absorb infrastructure boxing the LTS compiler can't elide. The
    # surface-API resolvers are designed for zero heap on Julia ≥ 1.12; the
    # struct-allocation cases also accept a small `+ ALLOC_THRESHOLD` slack on
    # LTS for the same reason.
    bc_no = NoBC()
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)
    x_vec = [0.0, 1.0, 2.0, 3.0]
    y = [10.0, 20.0, 30.0, 40.0]

    # Warmup so JIT compiles the specialized methods
    _resolve_axis(x_vec, bc_no)
    _resolve_axis(x_vec, bc_excl)
    _resolve_data(y, bc_no)
    _resolve_data(y, bc_excl)

    @testset "Vector + NoBC: zero alloc (passthrough)" begin
        @test (@allocated _resolve_axis(x_vec, bc_no)) <= ALLOC_THRESHOLD
        @test (@allocated _resolve_data(y, bc_no)) <= ALLOC_THRESHOLD
    end

    @testset "Vector + :exclusive: ≤ small alloc (one struct)" begin
        # `_ExclusivePeriodicAxis(x, period)` allocates the struct itself
        # (~24-64 bytes on 64-bit Julia) but does NOT copy `x`. Ditto for Data.
        # Acceptable for surface-API one-shot wrappers.
        a_axis = @allocated _resolve_axis(x_vec, bc_excl)
        a_data = @allocated _resolve_data(y, bc_excl)
        @test a_axis <= 64 + ALLOC_THRESHOLD
        @test a_data <= 64 + ALLOC_THRESHOLD
    end

    @testset "Range: stack-allocated _CachedRange (zero heap)" begin
        r = 0.0:1.0:3.0
        _resolve_axis(r, bc_no)       # warmup
        _resolve_axis(r, bc_excl)
        @test (@allocated _resolve_axis(r, bc_no)) <= ALLOC_THRESHOLD
        @test (@allocated _resolve_axis(r, bc_excl)) <= ALLOC_THRESHOLD
    end
end

@testitem "Base.copy(::_CachedVector) — deep-copy ownership contract" begin
    using FastInterpolations: _CachedVector

    # Lock down the contract that `Base.copy(::_CachedVector)` returns a
    # FULLY independent instance — every field gets its own fresh allocation.
    # The alias-h/inv_h optimization was explicitly rejected during PR review
    # (commit history: a25dbad33 → 5fb415ca7) to keep `copy`'s ownership
    # semantics clean. Regressing to aliased h/inv_h would silently couple
    # source and result lifetimes.
    x = [0.0, 1.0, 2.5, 4.0, 6.0]
    c = _CachedVector(x)

    @testset "All three buffers are independent" begin
        c2 = copy(c)
        @test c2 isa _CachedVector{Float64, Float64}
        @test c2.inner == c.inner
        @test c2.h == c.h
        @test c2.inv_h == c.inv_h
        @test c2.inner !== c.inner
        @test c2.h !== c.h
        @test c2.inv_h !== c.inv_h
    end

    @testset "Mutating source.inner doesn't affect copy" begin
        c2 = copy(c)
        c.inner[1] = 999.0
        @test c2.inner[1] != 999.0
    end

    @testset "Mutating source.h / inv_h doesn't affect copy" begin
        c2 = copy(c)
        c.h[1] = 999.0
        c.inv_h[1] = -999.0
        @test c2.h[1] != 999.0
        @test c2.inv_h[1] != -999.0
    end

    @testset "Mutating copy.inner doesn't affect source" begin
        c_fresh = _CachedVector([0.0, 1.0, 2.5, 4.0, 6.0])
        c2 = copy(c_fresh)
        c2.inner[1] = 777.0
        @test c_fresh.inner[1] != 777.0
    end
end

@testitem "Base.copy(::_ExclusivePeriodicAxis) — recursive deep copy" begin
    using FastInterpolations: _CachedVector, _ExclusivePeriodicAxis

    # `_ExclusivePeriodicAxis` wraps a `_CachedVector` (or other inner). Its
    # `Base.copy` must recurse into the inner buffer so the outer copy is
    # fully independent of the source. This is what `_convert_copy(g, Tg)`
    # delegates to in same-eltype 1D/ND inner ctors.
    x = [0.0, 1.0, 2.0, 3.0]
    cv = _CachedVector(x)
    g = _ExclusivePeriodicAxis(cv, 4.0)

    g2 = copy(g)
    @test g2 isa _ExclusivePeriodicAxis
    @test g2.inner !== g.inner                           # fresh inner wrapper
    @test g2.inner.inner !== g.inner.inner               # fresh user-buffer copy
    @test g2.period == g.period

    g.inner.inner[1] = 999.0
    @test g2.inner.inner[1] != 999.0                     # source-mutation isolated
end

@testitem "_cache_axis 3-arg Tg form: pre-wrapped + :exclusive coverage" begin
    using FastInterpolations:
        _cache_axis,
        _CachedRange, _CachedVector, _ExclusivePeriodicAxis,
        PeriodicBC, _to_float

    # The 3-arg `_cache_axis(x, bc, Tg)` overloads for pre-wrapped axes +
    # `:exclusive` BC are the missing rung in the canonical path: outer
    # factories produce raw → wrapped (one-shot wrap by the 3-arg `Vector`
    # / `Range` overload), so pre-wrapped + `:exclusive` only fires when
    # callers thread an already-wrapped axis through `_cache_axis(_, bc, Tg)`
    # explicitly. Exercise those rungs here so the 3-arg dispatch table is
    # uniformly covered.
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "_CachedRange + :exclusive (3-arg)" begin
        cr = _to_float(0.0:1.0:3.0, Float64)
        ax = _cache_axis(cr, bc_excl, Float64)
        @test ax isa _ExclusivePeriodicAxis
        @test ax.inner === cr
        @test ax.period == 4.0
    end

    @testset "_CachedVector + :exclusive (3-arg)" begin
        cv = _CachedVector([0.0, 1.0, 2.0, 3.0])
        ax = _cache_axis(cv, bc_excl, Float64)
        @test ax isa _ExclusivePeriodicAxis
        @test ax.inner === cv
        @test ax.period == 4.0
    end

    @testset "_ExclusivePeriodicAxis + :exclusive / NoBC (3-arg, idempotent)" begin
        cv = _CachedVector([0.0, 1.0, 2.0, 3.0])
        g = _ExclusivePeriodicAxis(cv, 4.0)
        @test _cache_axis(g, bc_excl, Float64) === g
        @test _cache_axis(g, FastInterpolations.NoBC(), Float64) === g
    end
end

@testitem "_convert_copy(::_ExclusivePeriodicAxis, ::Type{T2}) cross-eltype rebuild" begin
    using FastInterpolations: _CachedVector, _ExclusivePeriodicAxis, _convert_copy

    # Same-eltype path is exercised by `Base.copy` testitem above; this
    # exercises the cross-eltype rebuild branch (Float32 → Float64 here).
    cv32 = _CachedVector(Float32[0.0, 1.0, 2.0, 3.0])
    g32 = _ExclusivePeriodicAxis(cv32, 4.0f0)

    g64 = _convert_copy(g32, Float64)
    @test g64 isa _ExclusivePeriodicAxis
    @test eltype(g64) == Float64
    @test g64.inner isa _CachedVector{Float64}
    @test g64.inner !== g32.inner               # fresh wrapper
    @test g64.period == 4.0f0                   # period kept (its own type)
    @test collect(g64.inner.inner) == [0.0, 1.0, 2.0, 3.0]
end

# ─────────────────────────────────────────────────────────────────────────────
# Dedicated dispatch tests for `_cache_axis` 3-arg Tg form
# ─────────────────────────────────────────────────────────────────────────────
# These lock in the dispatch table documented in the "Tg-aware 3-arg overloads"
# section of `src/core/periodic_axis.jl`.
# They distinguish:
#   • RAW inputs (Vector/Range) → Tg respected via `_to_float`
#   • PRE-WRAPPED inputs (_CachedRange/_CachedVector/_ExclusivePeriodicAxis)
#     → Tg INTENTIONALLY ignored (passthrough; downstream `_convert_copy`
#     handles the eltype work in the canonical two-step pattern).

@testitem "_cache_axis 3-arg Tg respected: raw Vector (eltype mismatch)" begin
    using FastInterpolations: _cache_axis, _CachedVector, NoBC, PeriodicBC

    bc_no = NoBC()
    bc_inc = PeriodicBC(endpoint = :inclusive)
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    # Float32 grid promotes to Float64 via `_to_float` broadcasting.
    x32 = Float32[0.0, 1.0, 2.0, 3.0]

    @testset "Vector{Float32} + Tg=Float64 promotes via _to_float" begin
        c = _cache_axis(x32, bc_no, Float64)
        @test c isa _CachedVector{Float64}
        @test eltype(c) === Float64
        @test eltype(c.h) === Float64
        @test eltype(c.inv_h) === Float64
    end

    @testset "Vector{Float32} + :inclusive + Tg=Float64" begin
        c = _cache_axis(x32, bc_inc, Float64)
        @test c isa _CachedVector{Float64}
        @test eltype(c) === Float64
    end

    @testset "Vector{Float32} + :exclusive + Tg=Float64" begin
        ax = _cache_axis(x32, bc_excl, Float64)
        @test ax isa FastInterpolations._ExclusivePeriodicAxis
        @test eltype(ax.inner) === Float64           # inner promoted to Tg
        @test ax.inner isa _CachedVector{Float64}
    end
end

@testitem "_cache_axis 3-arg Tg respected: raw Range (eltype mismatch)" begin
    using FastInterpolations: _cache_axis, _CachedRange, _ExclusivePeriodicAxis, NoBC, PeriodicBC

    bc_no = NoBC()
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "Int range + Tg=Float32 → _CachedRange{Float32}" begin
        r = 0:1:3                                     # eltype Int
        cr = _cache_axis(r, bc_no, Float32)
        @test cr isa _CachedRange{Float32}
        @test eltype(cr) === Float32
    end

    @testset "Float64 range + Tg=Float32 → _CachedRange{Float32}" begin
        r = 0.0:1.0:3.0
        cr = _cache_axis(r, bc_no, Float32)
        @test cr isa _CachedRange{Float32}
    end

    @testset "Int range + :exclusive + Tg=Float64" begin
        r = 0:1:3
        ax = _cache_axis(r, bc_excl, Float64)
        @test ax isa _ExclusivePeriodicAxis
        @test ax.inner isa _CachedRange{Float64}     # inner promoted to Tg
    end

    @testset "Int range + :exclusive + Float64 period + Tg=Float32 stays Float32" begin
        # Convert-first contract: the period resolves against the Tg-typed axis,
        # so a Float64 period literal cannot re-widen the value-matched axis.
        r = 0:1:3
        ax = _cache_axis(r, bc_excl, Float32)
        @test ax isa _ExclusivePeriodicAxis
        @test ax.inner isa _CachedRange{Float32}
        @test ax.period isa Float32
    end
end

@testitem "_cache_axis 3-arg Tg INTENTIONALLY IGNORED: pre-wrapped inputs" begin
    using FastInterpolations:
        _cache_axis, _CachedRange, _CachedVector, _ExclusivePeriodicAxis,
        _to_float, NoBC, PeriodicBC

    # CONTRACT LOCK-IN: `_cache_axis(pre-wrapped, bc, Tg)` is a passthrough
    # regardless of `Tg` — the wrapper's eltype is preserved verbatim. The
    # canonical persistent pattern is `_convert_copy(_cache_axis(x, bc, Tg), Tg)`
    # which separates wrapping (this call) from eltype enforcement (`_convert_copy`).
    # If this passthrough behavior changes silently, every persistent inner ctor
    # that follows the two-step pattern would acquire a hidden double-conversion.
    bc_no = NoBC()
    bc_inc = PeriodicBC(endpoint = :inclusive)
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "_CachedRange{Float64} + Tg=Float32 → passthrough (still Float64)" begin
        cr64 = _to_float(0.0:1.0:3.0, Float64)
        out = _cache_axis(cr64, bc_no, Float32)
        @test out === cr64                            # exact identity
        @test out isa _CachedRange{Float64}
        @test eltype(out) === Float64                 # NOT Float32 — passthrough
    end

    @testset "_CachedVector{Float64} + Tg=Float32 → passthrough" begin
        cv64 = _CachedVector([0.0, 1.0, 2.0, 3.0])
        out = _cache_axis(cv64, bc_no, Float32)
        @test out === cv64                            # exact identity
        @test eltype(out) === Float64
    end

    @testset "_CachedVector{Float64} + :exclusive + Tg=Float32 wraps but keeps inner eltype" begin
        cv64 = _CachedVector([0.0, 1.0, 2.0, 3.0])
        ax = _cache_axis(cv64, bc_excl, Float32)
        @test ax isa _ExclusivePeriodicAxis
        @test ax.inner === cv64                       # inner is the SAME wrapper
        @test eltype(ax.inner) === Float64            # NOT Float32 — passthrough
    end

    @testset "_CachedRange{Float32} + :inclusive + Tg=Float64 → passthrough" begin
        cr32 = _to_float(0.0f0:1.0f0:3.0f0, Float32)
        out = _cache_axis(cr32, bc_inc, Float64)
        @test out === cr32
        @test eltype(out) === Float32                 # NOT Float64
    end

    @testset "_ExclusivePeriodicAxis + Tg → idempotent (Tg ignored)" begin
        cv32 = _CachedVector(Float32[0.0, 1.0, 2.0, 3.0])
        g = _ExclusivePeriodicAxis(cv32, 4.0f0)
        @test _cache_axis(g, bc_no, Float64) === g    # passthrough; eltype Float32 kept
        @test _cache_axis(g, bc_excl, Float64) === g
    end
end

@testitem "_cache_axis canonical two-step pattern enforces Tg" begin
    using FastInterpolations:
        _cache_axis, _convert_copy, _CachedVector, _CachedRange, _ExclusivePeriodicAxis,
        NoBC, PeriodicBC

    # Verify that `_convert_copy(_cache_axis(x, bc, Tg), Tg)` produces a
    # Tg-typed owned axis for BOTH raw and pre-wrapped inputs. This is the
    # contract that justifies `_cache_axis` 3-arg's passthrough-on-pre-wrapped
    # behavior — the eltype enforcement is delegated to `_convert_copy`.
    bc_no = NoBC()
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "Raw Vector{Float32} → Float64 owned _CachedVector" begin
        x32 = Float32[0.0, 1.0, 2.0, 3.0]
        xc = _convert_copy(_cache_axis(x32, bc_no, Float64), Float64)
        @test xc isa _CachedVector{Float64}
        @test eltype(xc) === Float64
        @test xc.inner !== x32                        # owned (independent buffer)
    end

    @testset "Pre-wrapped _CachedVector{Float32} → Float64 owned" begin
        cv32 = _CachedVector(Float32[0.0, 1.0, 2.0, 3.0])
        xc = _convert_copy(_cache_axis(cv32, bc_no, Float64), Float64)
        @test xc isa _CachedVector{Float64}
        @test eltype(xc) === Float64                  # `_convert_copy` enforced Tg
        @test xc !== cv32                             # fresh wrapper
    end

    @testset "Raw Int range → Float64 owned with :exclusive" begin
        r = 0:1:3
        ax = _convert_copy(_cache_axis(r, bc_excl, Float64), Float64)
        @test ax isa _ExclusivePeriodicAxis
        @test eltype(ax.inner) === Float64
    end

    @testset "Pre-wrapped _ExclusivePeriodicAxis{Float32} → Float64 owned" begin
        cv32 = _CachedVector(Float32[0.0, 1.0, 2.0, 3.0])
        g32 = _ExclusivePeriodicAxis(cv32, 4.0f0)
        g64 = _convert_copy(_cache_axis(g32, bc_excl, Float64), Float64)
        @test g64 isa _ExclusivePeriodicAxis
        @test eltype(g64) === Float64
        @test g64.inner.inner !== cv32.inner          # fresh user-buffer
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Dedicated dispatch tests for `_cache_axis_pooled` (one-shot variant)
# ─────────────────────────────────────────────────────────────────────────────
# These lock in the dispatch table documented at `cached_vector.jl` — the
# pool-backed twin of `_cache_axis`. Unlike persistent `_cache_axis`, the
# pooled 3-arg form does NOT have a downstream `_convert_copy` stage, so it
# threads `Tg` uniformly via `_to_float` (catch-all) for every input type.

@testitem "_cache_axis_pooled 2-arg dispatch table" begin
    using FastInterpolations: _cache_axis_pooled, _CachedRange, _CachedVector, _to_float
    using AdaptiveArrayPools: @with_pool

    # Assertions live inside the `@with_pool` scope so pool-owned `h`/`inv_h`
    # buffers never escape — recycled storage would otherwise let stale reads
    # slip in once a sibling test re-acquires the same slots.
    @with_pool pool function _smoke()
        x_vec = [0.0, 1.0, 2.5, 4.0]
        x_range = 0.0:1.0:3.0
        x_int_range = 0:1:3
        cr64 = _to_float(x_range, Float64)
        cv64 = _CachedVector([0.0, 1.0, 2.0, 3.0])

        cv_out = _cache_axis_pooled(pool, x_vec)
        cr_out = _cache_axis_pooled(pool, x_range)
        cr_int_out = _cache_axis_pooled(pool, x_int_range)
        cv_pass = _cache_axis_pooled(pool, cv64)
        cr_pass = _cache_axis_pooled(pool, cr64)

        @testset "Pool wrap / passthrough rules (no Tg)" begin
            # Vector{Float64} → pool-backed _CachedVector{Float64}
            @test cv_out isa _CachedVector{Float64}
            @test eltype(cv_out) === Float64
            @test length(cv_out.h) == 3                   # n-1
            @test cv_out.h ≈ [1.0, 1.5, 1.5]

            # Float Range → _CachedRange{Float64}
            @test cr_out isa _CachedRange{Float64}
            @test eltype(cr_out) === Float64

            # Int Range → _CachedRange{Float64} (float-defaulted via `float(eltype(x))`)
            @test cr_int_out isa _CachedRange{Float64}
            @test eltype(cr_int_out) === Float64

            # Pre-wrapped inputs round-trip unchanged (===)
            @test cv_pass === cv64
            @test cr_pass === cr64
        end
        return nothing
    end

    _smoke()
end

@testitem "_cache_axis_pooled 3-arg Tg respected: raw inputs" begin
    using FastInterpolations: _cache_axis_pooled, _CachedRange, _CachedVector
    using AdaptiveArrayPools: @with_pool

    # Assertions live inside the `@with_pool` scope so pool-owned `h`/`inv_h`
    # buffers never escape.
    @with_pool pool function _smoke()
        x32 = Float32[0.0, 1.0, 2.0, 3.0]              # raw Vector{Float32}
        r_int = 0:1:3                                  # Int range
        r_f64 = 0.0:1.0:3.0                            # Float64 range

        # Warm up compilation/pool before the allocation assertions below.
        _cache_axis_pooled(pool, x32, Float64)
        cv_promoted = _cache_axis_pooled(pool, x32, Float64)
        cr_int_promoted = _cache_axis_pooled(pool, r_int, Float32)
        cr_f64_demoted = _cache_axis_pooled(pool, r_f64, Float32)

        @testset "Vector{Float32} + Tg=Float64 → _CachedVector{Float64}" begin
            @test cv_promoted isa _CachedVector{Float64}
            @test eltype(cv_promoted) === Float64
        end

        @testset "Int range + Tg=Float32 → _CachedRange{Float32}" begin
            @test cr_int_promoted isa _CachedRange{Float32}
            @test eltype(cr_int_promoted) === Float32
        end

        @testset "Float64 range + Tg=Float32 → _CachedRange{Float32} (demotion)" begin
            @test cr_f64_demoted isa _CachedRange{Float32}
            @test eltype(cr_f64_demoted) === Float32
        end
        return nothing
    end

    _smoke()
end

@testitem "_cache_axis_pooled 3-arg Tg same-eltype identity passthrough" begin
    using FastInterpolations: _cache_axis_pooled, _CachedRange, _CachedVector, _to_float
    using AdaptiveArrayPools: @with_pool

    # When `Tg == eltype(x)`, the `_to_float` delegation hits the identity
    # `_to_float(::AbstractVector{T}, ::Type{T}) = x` overload (utils.jl:33),
    # so the 3-arg form behaves exactly like the 2-arg form — no conversion,
    # no warning, no extra allocation beyond the 2-arg path's pool work.
    # Assertions live inside the `@with_pool` scope so pool-owned `h`/`inv_h`
    # buffers never escape.
    @with_pool pool function _smoke()
        x64 = [0.0, 1.0, 2.0, 3.0]
        r64 = 0.0:1.0:3.0
        cv64 = _CachedVector([0.0, 1.0, 2.0, 3.0])
        cr64 = _to_float(r64, Float64)

        cv_a = _cache_axis_pooled(pool, x64, Float64)         # raw vector, same Tg
        cr_a = _cache_axis_pooled(pool, r64, Float64)         # raw range, same Tg
        cv_b = _cache_axis_pooled(pool, cv64, Float64)        # pre-wrapped vector, same Tg
        cr_b = _cache_axis_pooled(pool, cr64, Float64)        # pre-wrapped range, same Tg

        @test cv_a isa _CachedVector{Float64}
        @test cr_a isa _CachedRange{Float64}
        @test cv_b === cv64                                   # passthrough
        @test cr_b === cr64                                   # passthrough
        return nothing
    end

    _smoke()
end

@testitem "_cache_axis_pooled pool DATA-buffer reuse after warmup" setup = [AllocConstants] begin
    using FastInterpolations: _cache_axis_pooled, _CachedVector
    using AdaptiveArrayPools: @with_pool

    # CONTRACT: after warmup, `acquire!(pool, T, n-1)` REUSES the same pool
    # buffer for `h`/`inv_h` — no fresh n-sized `Vector{T}` per call. Pool
    # reuse happens ACROSS function invocations (cursor rewinds at scope
    # exit), so warmups are separate calls of `_measure_*`.
    #
    # JIT NOTE: `_measure_*` is defined ONCE at testitem scope so JIT runs
    # only on the first invocation. Don't wrap `@allocated` in another helper
    # — that wrapper's first call would pay its own JIT cost (~400 B).
    #
    # End-to-end zero-alloc of `quadratic_interp` is enforced by
    # `test_quadratic.jl`; this test isolates the pool-DATA contract for
    # `_cache_axis_pooled` standalone.

    # Wrapper aliases pool buffers — MUST NOT escape `@with_pool` scope
    # (CLAUDE.md "Pool Safety Rules"). Extract a scalar inside the scope.
    @with_pool pool function _measure_vec(x)
        c = _cache_axis_pooled(pool, x)
        return c.h[1] + c.inv_h[1]
    end
    @with_pool pool function _measure_range(r)
        c = _cache_axis_pooled(pool, r)
        return Float64(c[1])
    end

    @testset "Raw Vector (n=257) — pool DATA reused (no fresh n-sized Vector)" begin
        x = [(i - 1) * 1.0 for i in 1:257]
        _measure_vec(x)                  # warmup 1 (JIT + pool grow)
        _measure_vec(x)                  # warmup 2 (pool reuses)
        @test (@allocated _measure_vec(x)) <= ALLOC_THRESHOLD
    end

    @testset "Raw Range — `_CachedRange` is value-typed (no n-sized alloc)" begin
        r = 0.0:1.0:256.0
        _measure_range(r)
        _measure_range(r)
        @test (@allocated _measure_range(r)) <= ALLOC_THRESHOLD
    end
end

@testitem "_cache_axis_pooled — view input pool-acquires inner (no heap alloc)" setup = [AllocConstants] begin
    using FastInterpolations: _cache_axis_pooled, _CachedVector
    using AdaptiveArrayPools: @with_pool

    # `inner::Vector{T}` is forced for cache-key uniformity. Non-Vector input
    # (view, OffsetArray, etc.) is copied into a pool-acquired Vector instead
    # of `Vector{T}(x)`, so view input stays zero-heap after warmup.
    WRAPPER_OVERHEAD_LIMIT = 512

    @with_pool pool function _measure_view(vw)
        c = _cache_axis_pooled(pool, vw)
        return c.inner[1] + c.h[1] + c.inv_h[1]
    end

    @testset "View input → inner materialized to Vector{T} (single concrete type)" begin
        @with_pool pool function _check_type(vw)
            c = _cache_axis_pooled(pool, vw)
            return (c isa _CachedVector{Float64, Float64}, c.inner isa Vector{Float64})
        end
        big = [0.0, 1.0, 2.5, 4.0, 6.0, 9.0, 12.0, 15.0]
        is_concrete, inner_is_vector = _check_type(@view big[2:7])
        @test is_concrete
        @test inner_is_vector
    end

    @testset "View input — zero heap alloc after warmup (pool reuses inner)" begin
        big = [0.0, 1.0, 2.5, 4.0, 6.0, 9.0, 12.0, 15.0]
        vw = @view big[2:7]
        _measure_view(vw)                       # warmup 1 (pool grow)
        _measure_view(vw)                       # warmup 2 (pool reuse)
        @test (@allocated _measure_view(vw)) <= WRAPPER_OVERHEAD_LIMIT
    end
end

# RED PIN (#7 root / #1): the one-shot `_resolve_axis` :exclusive Vector arm must respect the
# value-matched `Tg` exactly as the persistent `_cache_axis` does. Today `_resolve_axis` resolves
# the period against the RAW grid (→ `float(Int)` = Float64) while `_cache_axis` converts-first
# (→ Float32), so an Int Vector + Tg=Float32 diverges: Float64 axis vs Float32 axis. That divergence
# IS the linear one-shot crash and the 2-arg/3-arg period-timing split — a `_wrap_exclusive`
# helper shared by both families (always convert-first) would close it.
@testitem "_resolve_axis :exclusive Vector respects Tg like _cache_axis (narrow-float root)" begin
    using FastInterpolations: _resolve_axis, _cache_axis, PeriodicBC

    bc = PeriodicBC(endpoint = :exclusive, period = 2.5)
    x = collect(0:2)                                   # Int Vector; values exact in Float32

    axr = _resolve_axis(x, bc, Float32)                # one-shot arm
    axc = _cache_axis(x, bc, Float32)                  # persistent arm (reference; already narrow)

    @test eltype(axc) === Float32                       # sanity: persistent narrows correctly
    @test eltype(axr) === Float32                       # RED: one-shot currently Float64
    @test axr.period isa Float32                        # RED: one-shot currently Float64
    @test eltype(axr) === eltype(axc)                   # one-shot ≡ persistent axis width
end

# RED PIN (Copilot #182): `_resolve_axis` only has the 1-arg `_ExclusivePeriodicAxis` passthrough
# (periodic_axis.jl), so the 2-arg/3-arg `:exclusive` forms fall through to the AbstractVector arms
# and RE-WRAP an already-wrapped axis (`_ExclusivePeriodicAxis <: AbstractVector`) — nesting toward
# length (n+1)+1, which actually throws in the ctor (`inner[end] < x_max` fails). `_cache_axis`
# already defends this with a full passthrough set; `_resolve_axis` must mirror it.
@testitem "_resolve_axis :exclusive passes through a pre-wrapped axis (no re-wrap)" begin
    using FastInterpolations: _resolve_axis, _wrap_exclusive, _ExclusivePeriodicAxis, PeriodicBC

    bc = PeriodicBC(endpoint = :exclusive, period = 4.0)
    wrapped = _wrap_exclusive(collect(0.0:3.0), bc)     # _ExclusivePeriodicAxis, length n+1 = 5
    @test wrapped isa _ExclusivePeriodicAxis
    @test length(wrapped) == 5

    # Feeding the already-wrapped axis back must PASS THROUGH (identity), not nest/throw.
    @test _resolve_axis(wrapped, bc) === wrapped                 # 2-arg
    @test _resolve_axis(wrapped, bc, Float64) === wrapped        # 3-arg Tg-aware
    @test length(_resolve_axis(wrapped, bc, Float64)) == 5       # not (n+1)+1
end

# Width-first geometry primitives: `_get_*(Tw, x, i)` — the value-matched coordinate
# width `Tw` comes from the caller's surface (`_promote_grid_float(Tg, Tv)`). Raw axes
# difference in their OWN eltype first (Int spans are exact), convert the SPAN once,
# then divide — the reciprocal is BORN at `Tw` (no `inv(Int)::Float64` minting), and
# coordinates beyond `Tw`'s ulp cannot cancel (span-first, never endpoint-convert).
# Wrapped axes reuse the cached reciprocal (convert is a no-op once value-matched).
# Width-less forms keep the historic raw-eltype behavior via delegation.
@testitem "width-first _get_h/_get_inv_h/_get_inv_2cell + secants" begin
    using FastInterpolations: _get_h, _get_inv_h, _get_inv_2cell,
        _forward_secant, _backward_secant, _centered_secant,
        _cache_axis, _resolve_axis, NoBC

    # raw Int vector: reciprocal born at Tw
    x = [1, 3, 6]
    @test _get_h(Float32, x, 1) === 2.0f0
    @test _get_inv_h(Float32, x, 1) === 0.5f0
    @test _get_inv_2cell(Float32, x, 2) === inv(5.0f0)
    @test _get_inv_h(Float64, x, 2) === inv(3.0)

    # span-first precision: Int coords beyond Float32's ulp — endpoint-convert
    # would cancel to 0 (inv → Inf); the span itself is small and exact.
    xb = [16_777_216, 16_777_218]
    @test _get_inv_h(Float32, xb, 1) === 0.5f0

    # wrapped axes: cached reciprocal reused; convert no-op when value-matched
    c = _cache_axis(collect(Float32, 1:5), NoBC())
    @test _get_inv_h(Float32, c, 2) === _get_inv_h(c, 2)
    r = _resolve_axis(1:5, Float32)                    # _CachedRange{Float32}
    @test _get_inv_h(Float32, r, 1) === 1.0f0
    @test _get_h(Float32, r, 1) === 1.0f0

    # width-first secants: Int axis + F32 data → Float32 end to end
    y32 = Float32[1, 2, 4]
    @test _forward_secant(Float32, x, y32, 1) === 0.5f0
    @test _backward_secant(Float32, x, y32, 2) === 0.5f0
    @test _centered_secant(Float32, x, y32, 2) === 0.6f0
    # width-less forms keep the historic raw semantics (Int axis → Float64)
    @test _forward_secant(x, y32, 1) isa Float64
end
