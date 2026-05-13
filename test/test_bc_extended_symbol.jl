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
