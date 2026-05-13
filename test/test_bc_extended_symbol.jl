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

@testitem "PeriodicBC :extended — trait orthogonality" begin
    using FastInterpolations: PeriodicBC, NoBC
    using FastInterpolations: _is_periodic_inclusive, _is_periodic_seam_folded

    bc_inc = PeriodicBC(endpoint = :inclusive)
    bc_exc = PeriodicBC(endpoint = :exclusive, period = 2π)
    bc_ext = PeriodicBC{:extended, Float64, false}(2π)
    no_bc  = NoBC()

    # _is_periodic_inclusive: true for :inclusive and :extended only
    @test !_is_periodic_inclusive(no_bc)
    @test  _is_periodic_inclusive(bc_inc)
    @test !_is_periodic_inclusive(bc_exc)
    @test  _is_periodic_inclusive(bc_ext)

    # _is_periodic_seam_folded: true for :exclusive and :extended only
    @test !_is_periodic_seam_folded(no_bc)
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
    bc_inc       = PeriodicBC(endpoint = :inclusive)
    bc_inc_nochk = PeriodicBC(endpoint = :inclusive, check = false)
    @test _bc_after_extend(bc_inc)       === bc_inc
    @test _bc_after_extend(bc_inc_nochk) === bc_inc_nochk

    # Non-periodic → passthrough
    nb = NoBC()
    @test _bc_after_extend(nb) === nb
end

@testitem "PeriodicBC :extended — cascade to Hermite-family 1D forward" begin
    using FastInterpolations: PeriodicBC
    n = 8
    period = 2π
    x = collect(range(0.0, step = period/n, length = n))
    y = sin.(x)
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    for build in (pchip_interp, cardinal_interp, akima_interp)
        itp = build(x, y; bc = bc_exc)
        # Internal axis was extended to length n+1
        @test length(itp.x) == n + 1
        # Forward eval correctness preserved (extension + bc symbol change
        # are introspection-only; numerical path is unchanged).
        @test itp(x[1])              ≈ y[1] atol = 1e-10
        @test itp(x[1] + period)     ≈ y[1] atol = 1e-10
    end
end

@testitem "PeriodicBC :extended — _adjoint_user_n_axis" begin
    using FastInterpolations: PeriodicBC, NoBC
    using FastInterpolations: _adjoint_user_n_axis

    # :inclusive — layout already user-sized → no shrink
    @test _adjoint_user_n_axis(PeriodicBC(endpoint = :inclusive), 9) == 9
    # :exclusive — OneShot wrap layout → trim
    @test _adjoint_user_n_axis(PeriodicBC(endpoint = :exclusive, period = 2π), 9) == 8
    # :extended — promoted layout → trim
    @test _adjoint_user_n_axis(PeriodicBC{:extended, Float64, false}(2π), 9) == 8
    # Non-periodic — no shrink
    @test _adjoint_user_n_axis(NoBC(), 9) == 9
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
    x = collect(range(0.0, step = period/n, length = n))
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
    x = collect(range(0.0, step = period/n, length = n))
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
    x1 = collect(range(0.0, step = 2π/n1, length = n1))
    x2 = collect(range(0.0, step = 2π/n2, length = n2))
    bc2t = (PeriodicBC(endpoint = :exclusive, period = 2π),
            PeriodicBC(endpoint = :exclusive, period = 2π))
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
    x = collect(range(0.0, step = period/n, length = n))
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
    x = collect(range(0.0, step = period/n, length = n))
    y = sin.(x)
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    for build in (linear_interp, constant_interp)
        itp = build(x, y; bc = bc_exc)
        if build === linear_interp
            for i in 1:n
                @test itp(x[i]) ≈ y[i] atol = 1e-12
            end
        end
        for xq in (0.3, 1.4, 2.7, 4.2)
            @test itp(xq + period) ≈ itp(xq) atol = 1e-12
            @test itp(xq - period) ≈ itp(xq) atol = 1e-12
        end
    end
end

@testitem "PeriodicBC :extended — show methods" begin
    using FastInterpolations: PeriodicBC
    n = 8; period = 2π
    x = collect(range(0.0, step = period/n, length = n))
    y = sin.(x)
    itp = cubic_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = period))

    # Compact MIME"text/plain" display of the interpolant annotates `:extended`.
    s = sprint(show, MIME"text/plain"(), itp)
    @test occursin("Periodic", s)
    @test occursin("extended", s)
end
