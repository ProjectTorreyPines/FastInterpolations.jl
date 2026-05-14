@testitem "PeriodicBC :extended — type construction" begin
    using FastInterpolations: PeriodicBC

    # Direct internal-form construction works
    bc = PeriodicBC{:extended, Float64, false}(2π)
    @test bc isa PeriodicBC{:extended, Float64, false}
    @test bc.period == 2π

    # Validation: invalid endpoint symbol is rejected by inner constructor
    @test_throws Exception PeriodicBC{:bogus, Nothing, true}(nothing)
end

@testitem "PeriodicBC :extended — user kwarg constructor rejects" begin
    using FastInterpolations: PeriodicBC

    # User-facing keyword constructor must NOT accept :extended
    @test_throws ArgumentError PeriodicBC(endpoint = :extended)
    @test_throws ArgumentError PeriodicBC(endpoint = :extended, period = 2π)
end

@testitem "PeriodicBC :extended — seam-fold trait" begin
    using FastInterpolations: PeriodicBC, NoBC
    using FastInterpolations: _is_periodic_seam_folded

    bc_inc = PeriodicBC(endpoint = :inclusive)
    bc_exc = PeriodicBC(endpoint = :exclusive, period = 2π)
    bc_ext = PeriodicBC{:extended, Float64, false}(2π)

    # true for :exclusive and :extended only
    @test !_is_periodic_seam_folded(NoBC())
    @test !_is_periodic_seam_folded(bc_inc)
    @test  _is_periodic_seam_folded(bc_exc)
    @test  _is_periodic_seam_folded(bc_ext)
end

@testitem "PeriodicBC :extended — _bc_after_extend keystone" begin
    using FastInterpolations: PeriodicBC, NoBC
    using FastInterpolations: _bc_after_extend

    # :exclusive → :extended, period preserved, check pinned to false
    bc_exc = PeriodicBC(endpoint = :exclusive, period = 2π)
    bc_out = _bc_after_extend(bc_exc)
    @test bc_out isa PeriodicBC{:extended, Float64, false}
    @test bc_out.period == 2π

    # :inclusive → unchanged (preserves user's check flag)
    bc_inc = PeriodicBC(endpoint = :inclusive)
    bc_inc_nochk = PeriodicBC(endpoint = :inclusive, check = false)
    @test _bc_after_extend(bc_inc) === bc_inc
    @test _bc_after_extend(bc_inc_nochk) === bc_inc_nochk

    # Non-periodic → passthrough
    nb = NoBC()
    @test _bc_after_extend(nb) === nb
end

@testitem "PeriodicBC :extended — cascade to Hermite-family 1D forward" begin
    using FastInterpolations: PeriodicBC
    n = 8
    period = 2π
    x = collect(range(0.0, step = period / n, length = n))
    y = sin.(x)
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    for build in (pchip_interp, cardinal_interp, akima_interp)
        itp = build(x, y; bc = bc_exc)
        # Internal axis was extended to length n+1
        @test length(itp.x) == n + 1
        # Forward eval correctness preserved (extension + bc symbol change
        # are introspection-only; numerical path is unchanged).
        @test itp(x[1]) ≈ y[1] atol = 1.0e-10
        @test itp(x[1] + period) ≈ y[1] atol = 1.0e-10
    end
end

@testitem "PeriodicBC :extended — _has_seam_fold renamed predicate" begin
    using FastInterpolations: PeriodicBC, NoBC
    using FastInterpolations: _has_seam_fold

    bc_inc = PeriodicBC(endpoint = :inclusive)
    bc_exc = PeriodicBC(endpoint = :exclusive, period = 2π)
    bc_ext = PeriodicBC{:extended, Float64, false}(2π)

    @test !_has_seam_fold((NoBC(), bc_inc))
    @test  _has_seam_fold((NoBC(), bc_exc))
    @test  _has_seam_fold((NoBC(), bc_ext))
    @test  _has_seam_fold((bc_ext, bc_inc, bc_exc))
end

@testitem "PeriodicBC :extended — Cubic 1D forward introspection" begin
    using FastInterpolations: PeriodicBC
    n = 8; period = 2π
    x = collect(range(0.0, step = period / n, length = n))
    y = sin.(x)
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    itp = cubic_interp(x, y; bc = bc_exc)
    @test itp.bc isa PeriodicBC{:extended, Float64, false}
    @test itp.bc.period ≈ period
    @test length(itp.cache.x) == n + 1
end

@testitem "PeriodicBC :extended — Cubic 1D adjoint introspection" begin
    using FastInterpolations: PeriodicBC
    n = 8; period = 2π
    x = collect(range(0.0, step = period / n, length = n))
    xq = [0.3, 1.4, 2.7]
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    adj = cubic_adjoint(x, xq; bc = bc_exc)
    @test adj.bc isa PeriodicBC{:extended, Float64, false}
    @test length(adj.cache.x) == n + 1

    # Output size invariant: y_bar is length n (user dim)
    e = randn(length(xq))
    y_bar = adj(e)
    @test length(y_bar) == n
end

@testitem "PeriodicBC :extended — Cubic ND adjoint introspection" begin
    using FastInterpolations: PeriodicBC
    n1, n2 = 8, 10
    x1 = collect(range(0.0, step = 2π / n1, length = n1))
    x2 = collect(range(0.0, step = 2π / n2, length = n2))
    bc2t = (
        PeriodicBC(endpoint = :exclusive, period = 2π),
        PeriodicBC(endpoint = :exclusive, period = 2π),
    )
    xq = ([0.3, 1.4], [0.2, 2.7])

    adj = cubic_adjoint((x1, x2), xq; bc = bc2t)
    @test all(b -> b isa PeriodicBC{:extended, Float64, false}, adj.bcs)

    e = randn(length(xq[1]))
    y_bar = adj(e)
    @test size(y_bar) == (n1, n2)
end

@testitem "PeriodicBC :extended — Linear/Constant 1D forward introspection" begin
    using FastInterpolations: PeriodicBC, _CachedVector, _CachedRange,
        _ExclusivePeriodicAxis, _ExclusivePeriodicData
    n = 8; period = 2π
    x = collect(range(0.0, step = period / n, length = n))
    y = sin.(x)
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    # Note: LinearInterpolant / ConstantInterpolant don't carry a `bc` field
    # (pre-existing BC-Field-Unification gap). Introspection here verifies the
    # post-migration axis/data layout (length n+1, no wrappers, closed seam).
    for build in (linear_interp, constant_interp)
        itp = build(x, y; bc = bc_exc)
        @test length(itp.x) == n + 1
        @test !(itp.x isa _ExclusivePeriodicAxis)
        @test length(itp.y) == n + 1
        @test !(itp.y isa _ExclusivePeriodicData)
        @test itp.y[end] == itp.y[1]
    end
end

@testitem "PeriodicBC :extended — Linear/Constant 1D forward correctness" begin
    using FastInterpolations: PeriodicBC
    n = 8; period = 2π
    x = collect(range(0.0, step = period / n, length = n))
    y = sin.(x)
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    for build in (linear_interp, constant_interp)
        itp = build(x, y; bc = bc_exc)
        if build === linear_interp
            for i in 1:n
                @test itp(x[i]) ≈ y[i] atol = 1.0e-12
            end
        end
        for xq in (0.3, 1.4, 2.7, 4.2)
            @test itp(xq + period) ≈ itp(xq) atol = 1.0e-12
            @test itp(xq - period) ≈ itp(xq) atol = 1.0e-12
        end
    end
end

@testitem "PeriodicBC :extended — show methods" begin
    using FastInterpolations: PeriodicBC
    n = 8; period = 2π
    x = collect(range(0.0, step = period / n, length = n))
    y = sin.(x)
    itp = cubic_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = period))

    # Compact MIME"text/plain" display of the interpolant annotates `:extended`.
    s = sprint(show, MIME"text/plain"(), itp)
    @test occursin("Periodic", s)
    @test occursin("extended", s)
end

@testitem "PeriodicBC :extended — show methods, period === nothing branch" begin
    using FastInterpolations: PeriodicBC
    using FastInterpolations: _format_bc, _short_bc_name

    # Internal-form construction with `period === nothing` (mirrors the
    # pre-resolution state inside `_with_resolved_period` / inferred-period
    # call sites). Exercises the `period === nothing` branch of both formatters.
    bc_no_period = PeriodicBC{:extended, Nothing, false}(nothing)
    @test _format_bc(bc_no_period) == "Periodic (extended from :exclusive)"
    @test _short_bc_name(bc_no_period) == "Periodic(ext)"
end

@testitem "PeriodicBC :extended — Hermite-family 1D adjoint introspection" begin
    using FastInterpolations: PeriodicBC
    n = 8; period = 2π
    x = collect(range(0.0, step = period / n, length = n))
    y = sin.(x)
    xq = [0.3, 1.4, 2.7]
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    # PCHIP / Akima adjoints take `(x, y, xq)` — slopes are y-dependent.
    # Cardinal adjoint takes `(x, xq)` only (tension-parameterized, y-free).
    # All three should symmetrically promote the user's `:exclusive` to
    # `:extended` (matches `cubic_adjoint` symmetry).
    for adj in (
            pchip_adjoint(x, y, xq; bc = bc_exc),
            akima_adjoint(x, y, xq; bc = bc_exc),
            cardinal_adjoint(x, xq; bc = bc_exc),
        )
        @test adj.bc isa PeriodicBC{:extended, Float64, false}
        @test adj.bc.period ≈ period
        # Internal grid is extended length n+1; user-dim output is length n.
        @test length(adj.grid) == n + 1
        e = randn(length(xq))
        f_bar = adj(e)
        @test length(f_bar) == n
    end
end

@testitem "PeriodicBC :extended — _bc_after_extend is idempotent on :extended input" begin
    using FastInterpolations: PeriodicBC
    using FastInterpolations: _bc_after_extend

    # rrule-replay and double-call safety: applying `_bc_after_extend` to an
    # already-extended BC must be a no-op (else cascaded internal calls — see
    # `cubic_nd_adjoint.jl:559,579` — could silently change behavior).
    bc_ext = PeriodicBC{:extended, Float64, false}(2π)
    @test _bc_after_extend(bc_ext) === bc_ext

    # period === nothing variant
    bc_ext_no_period = PeriodicBC{:extended, Nothing, false}(nothing)
    @test _bc_after_extend(bc_ext_no_period) === bc_ext_no_period
end

@testitem "PeriodicBC :extended — _cache_axis with :extended bc returns plain wrap" begin
    using FastInterpolations: PeriodicBC, _CachedVector, _CachedRange,
        _ExclusivePeriodicAxis
    using FastInterpolations: _cache_axis, _convert_copy

    n = 8; period = 2π
    # Pre-extended length-(n+1) buffer (what `_periodic_extend_1d` produces).
    x_ext = collect(range(0.0, step = period / n, length = n + 1))
    bc_ext = PeriodicBC{:extended, Float64, false}(period)

    # `_cache_axis(x, ::PeriodicBC{:extended})` falls through to the
    # `::AbstractBC` overload — produces a plain `_CachedVector` (no
    # `_ExclusivePeriodicAxis` wrap, because the grid is already closed-cycle).
    wrapped = _cache_axis(x_ext, bc_ext)
    @test wrapped isa _CachedVector
    @test !(wrapped isa _ExclusivePeriodicAxis)
    @test length(wrapped) == n + 1

    # Canonical owned-copy pattern: `_cache_axis(_convert_copy(x, T), bc_ext)`.
    # `_convert_copy` makes a fresh buffer; `_cache_axis` aliases it.
    owned = _cache_axis(_convert_copy(x_ext, Float64), bc_ext)
    @test owned isa _CachedVector
    @test length(owned) == n + 1
    @test eltype(owned) === Float64

    # Range input — `_cache_axis(range, ::PeriodicBC{:extended})` also falls
    # through to `::AbstractBC` (plain `_to_float`, no wrapper).
    r = range(0.0, step = period / n, length = n + 1)
    rwrapped = _cache_axis(r, bc_ext)
    @test !(rwrapped isa _ExclusivePeriodicAxis)
    @test length(rwrapped) == n + 1
end

@testitem "PeriodicBC :extended — Hetero ND with Linear axis cascade" begin
    using FastInterpolations: PeriodicBC
    using FastInterpolations: _ExclusivePeriodicAxis

    nx, ny = 12, 10
    # User-dim n-point grid; library internally extends to n+1 inside hetero.
    x = collect(range(0.0, step = 2π / nx, length = nx))
    y = range(0.0, 1.0, ny)
    data = [sin(xi) * yj for xi in x, yj in y]

    bc_x = PeriodicBC(endpoint = :exclusive, period = 2π)
    methods = (LinearInterp(bc = bc_x), CubicInterp(bc = ZeroSlopeBC()))

    # Forward — verify wrap-around correctness through the unified API. This
    # exercises the per-axis `_cache_axis(g, bc_eff, Tg)` cascade for Linear
    # axes; with the `:extended` symbol present, the BC `bc_eff` produced by
    # `_bc_after_extend` selects the plain-wrap path (no `_ExclusivePeriodicAxis`
    # double-wrap when the cascade already extended the buffer).
    itp = interp((x, y), data; method = methods)

    # Eval inside the seam region [x[end], x[1]+period). Linear interpolation
    # between `data[end, j]` and `data[1, j]` (the closed seam point).
    xq_wrap = x[end] + 0.34
    yq = y[5]
    v_wrap = itp((xq_wrap, yq))
    α = (xq_wrap - x[end]) / (2π - x[end])
    v_expected = (1 - α) * (sin(x[end]) * yq) + α * (sin(x[1]) * yq)
    @test v_wrap ≈ v_expected atol = 1.0e-10

    # User-contract preservation: `methods[d].bc` retains `:exclusive` (rrule-
    # replay safety — see `hetero_adjoint.jl:654-658`). The promotion to
    # `:extended` lives in the internal cache/grid path, not in the user-facing
    # method tag.
    @test methods[1].bc isa PeriodicBC{:exclusive}
end
