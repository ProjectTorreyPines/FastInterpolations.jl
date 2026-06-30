# Coverage for the `_UnitStep` axis tag: which inputs earn it, the `Tinv`
# reciprocal-type contract, the accessor fold invariant, and that persistent
# interpolants (1D + ND) carry the intended tag — for both `_UnitStep` and
# `_Generic`.

@testitem "Axis tag — resolver assigns _UnitStep iff AbstractUnitRange" begin
    using FastInterpolations:
        _resolve_axis, _cache_axis, _to_float,
        _CachedRange, _UnitStep, _Generic, _get_h, _get_inv_h, NoBC

    tagof(x::_CachedRange) = typeof(x).parameters[3]

    @testset "AbstractUnitRange → _UnitStep (one-shot and persistent agree)" begin
        for g in (1:100, Base.OneTo(100))
            @test _resolve_axis(g) isa _CachedRange{Float64, Float64, _UnitStep}
            @test _cache_axis(g, NoBC(), Float64) isa _CachedRange{Float64, Float64, _UnitStep}
        end
    end

    @testset "Non-unit-range (incl. step-1 StepRange) → not _UnitStep" begin
        # `1:1:100` is a StepRange (never TwicePrecision) → _Generic on every arch.
        @test tagof(_resolve_axis(1:1:100)) === _Generic
        @test tagof(_cache_axis(1:1:100, NoBC(), Float64)) === _Generic
        # Float StepRangeLen is _Generic on aarch64; the x86_64 TwicePrecision fast
        # path tags it _WidenedDomain. Either way it is never _UnitStep, which is what
        # this testset pins (the concrete widen tag is covered in test_widendomain_tag).
        for g in (1.0:100.0, 0.0:0.5:50.0, range(0.0, 1.0; length = 100))
            @test tagof(_resolve_axis(g)) !== _UnitStep
            @test tagof(_cache_axis(g, NoBC(), Float64)) !== _UnitStep
        end
    end

    @testset "Tinv contract — reciprocal type, never the coordinate Int" begin
        # `Tinv == typeof(inv(oneunit(T)))`: Float64 for Int, T for Float. A
        # `_UnitStep` Int grid must NOT collapse to `_CachedRange{Int, Int}`.
        cr_i = _to_float(1:4, Int)
        @test cr_i isa _CachedRange{Int, Float64, _UnitStep}
        @test cr_i.h === 1                    # coordinate spacing stays Int
        @test cr_i.inv_h === 1.0              # reciprocal is Float64
        cr32 = _to_float(1:4, Float32)
        @test cr32 isa _CachedRange{Float32, Float32, _UnitStep}
        @test cr32.inv_h === 1.0f0
    end

    @testset "Accessors fold to one(Tinv) on _UnitStep; return field on _Generic" begin
        u = _to_float(1:4, Float64)           # _UnitStep — h ≡ inv_h ≡ 1
        @test _get_h(u) === 1.0 && _get_inv_h(u) === 1.0
        @test _get_h(u, 1) === 1.0 && _get_inv_h(u, 1) === 1.0                  # 2-arg delegates
        @test _get_h(u, 1, 0.0, 0.0) === 1.0 && _get_inv_h(u, 1, 0.0, 0.0) === 1.0  # 4-arg delegates
        g = _to_float(0.0:0.5:5.0, Float64)   # _Generic — non-unit step
        @test tagof(g) === _Generic
        @test _get_h(g) === 0.5 && _get_inv_h(g) === 2.0
    end
end

@testitem "Axis tag — survives into persistent interpolants (1D + ND)" begin
    using FastInterpolations: _CachedRange, _UnitStep, _Generic
    tagof(x::_CachedRange) = typeof(x).parameters[3]

    y = collect(Float64, 1:60) .^ 2
    data = [Float64(i + j) for i in 1:60, j in 1:60]

    @testset "1D — UnitRange → _UnitStep, Float range → _Generic" begin
        # grid field path differs by method: linear/quad/pchip `.x`, cubic `.cache.x`
        @test tagof(linear_interp(1:60, y).x) === _UnitStep
        @test tagof(linear_interp(1.0:60.0, y).x) === _Generic
        @test tagof(cubic_interp(1:60, y).cache.x) === _UnitStep
        @test tagof(cubic_interp(1.0:60.0, y).cache.x) === _Generic
        @test tagof(quadratic_interp(1:60, y).x) === _UnitStep
        @test tagof(pchip_interp(1:60, y).x) === _UnitStep
    end

    @testset "Constant keeps Tg=Int, _UnitStep, Float64 reciprocal" begin
        c = constant_interp(1:4, Float32[1, 4, 9, 16])
        @test c.x isa _CachedRange{Int, Float64, _UnitStep}
    end

    @testset "ND — per-axis tags are independent (mixed _UnitStep/_Generic)" begin
        itp = linear_interp((1:60, 1.0:60.0), data)
        @test tagof(itp.grids[1]) === _UnitStep
        @test tagof(itp.grids[2]) === _Generic
        itp2 = linear_interp((1:60, Base.OneTo(60)), data)
        @test all(tagof.(itp2.grids) .=== _UnitStep)
    end
end
