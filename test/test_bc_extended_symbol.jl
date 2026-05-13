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
