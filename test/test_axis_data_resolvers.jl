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
