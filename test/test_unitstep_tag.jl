# Coverage for the `_UnitStep`/`_OneTo` axis tags: which inputs earn them, the `Tinv`
# reciprocal-type contract, the accessor fold invariant, and that persistent
# interpolants (1D + ND) carry the intended tag. A unit range earns `_UnitStep`,
# `Base.OneTo` earns `_OneTo` (statically 1-based); a Float range is non-unit-step
# (`_Generic` on aarch64, `_WidenedDomain` on the x86_64 TwicePrecision fast path),
# so its assertions check `!== _UnitStep`.

@testitem "Axis tag — resolver assigns _UnitStep/_OneTo iff AbstractUnitRange/OneTo" begin
    using FastInterpolations:
        _resolve_axis, _cache_axis, _to_float,
        _CachedRange, _UnitStep, _OneTo, _Generic, _get_h, _get_inv_h, NoBC

    tagof(x::_CachedRange) = typeof(x).parameters[3]

    @testset "UnitRange → _UnitStep, Base.OneTo → _OneTo (one-shot and persistent agree)" begin
        @test _resolve_axis(1:100) isa _CachedRange{Float64, Float64, _UnitStep}
        @test _cache_axis(1:100, NoBC(), Float64) isa _CachedRange{Float64, Float64, _UnitStep}
        @test _resolve_axis(Base.OneTo(100)) isa _CachedRange{Float64, Float64, _OneTo}
        @test _cache_axis(Base.OneTo(100), NoBC(), Float64) isa _CachedRange{Float64, Float64, _OneTo}
        # `_OneTo` carries the same geometry as the equivalent `1:n` `_UnitStep` axis.
        @test _resolve_axis(Base.OneTo(100)).lo === 1.0
        @test collect(_resolve_axis(Base.OneTo(100))) == collect(_resolve_axis(1:100))
    end

    @testset "Non-unit-range (incl. step-1 StepRange) → not _UnitStep" begin
        # `1:1:100` is a StepRange (never TwicePrecision) → _Generic on every arch.
        @test tagof(_resolve_axis(1:1:100)) === _Generic
        @test tagof(_cache_axis(1:1:100, NoBC(), Float64)) === _Generic
        # Float StepRangeLen is _Generic on aarch64; the x86_64 TwicePrecision fast
        # path tags it _WidenedDomain. Either way it is never _UnitStep, which is what
        # this testset pins (the concrete widen tag is covered in test_widened_domain_tag).
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

    @testset "Accessors fold to one(Tinv) on _UnitStep/_OneTo, return the field otherwise" begin
        o = _to_float(Base.OneTo(4), Float64) # _OneTo — same folds as _UnitStep
        @test _get_h(o) === 1.0 && _get_inv_h(o) === 1.0
        u = _to_float(1:4, Float64)           # _UnitStep — h ≡ inv_h ≡ 1
        @test _get_h(u) === 1.0 && _get_inv_h(u) === 1.0
        @test _get_h(u, 1) === 1.0 && _get_inv_h(u, 1) === 1.0                  # 2-arg delegates
        @test _get_h(u, 1, 0.0, 0.0) === 1.0 && _get_inv_h(u, 1, 0.0, 0.0) === 1.0  # 4-arg delegates
        # Non-unit-step Float range: _Generic on aarch64, _WidenedDomain on x86_64 —
        # either way not _UnitStep, so the accessors return the field (no fold).
        g = _to_float(0.0:0.5:5.0, Float64)
        @test tagof(g) !== _UnitStep
        @test _get_h(g) === 0.5 && _get_inv_h(g) === 2.0
    end
end

@testitem "Axis tag — survives into persistent interpolants (1D + ND)" begin
    using FastInterpolations: _CachedRange, _UnitStep, _OneTo, _Generic
    tagof(x::_CachedRange) = typeof(x).parameters[3]

    y = collect(Float64, 1:60) .^ 2
    data = [Float64(i + j) for i in 1:60, j in 1:60]

    @testset "1D — UnitRange → _UnitStep, Float range → non-_UnitStep" begin
        # grid field path differs by method: linear/quad/pchip `.x`, cubic `.cache.x`.
        # A Float StepRangeLen is _Generic on aarch64 / _WidenedDomain on x86_64; the
        # invariant that survives the build is "not _UnitStep" (the exact x86 widen
        # tag is pinned in test_widened_domain_tag.jl).
        @test tagof(linear_interp(1:60, y).x) === _UnitStep
        @test tagof(linear_interp(1.0:60.0, y).x) !== _UnitStep
        @test tagof(cubic_interp(1:60, y).cache.x) === _UnitStep
        @test tagof(cubic_interp(1.0:60.0, y).cache.x) !== _UnitStep
        @test tagof(quadratic_interp(1:60, y).x) === _UnitStep
        @test tagof(pchip_interp(1:60, y).x) === _UnitStep
    end

    @testset "Constant keeps Tg=Int, _UnitStep, Float64 reciprocal" begin
        c = constant_interp(1:4, Float32[1, 4, 9, 16])
        @test c.x isa _CachedRange{Int, Float64, _UnitStep}
    end

    @testset "ND — per-axis tags are independent (UnitRange _UnitStep, Float non-_UnitStep)" begin
        itp = linear_interp((1:60, 1.0:60.0), data)
        @test tagof(itp.grids[1]) === _UnitStep
        @test tagof(itp.grids[2]) !== _UnitStep        # Float axis: _Generic / _WidenedDomain (x86)
        itp2 = linear_interp((1:60, Base.OneTo(60)), data)
        @test tagof(itp2.grids[1]) === _UnitStep
        @test tagof(itp2.grids[2]) === _OneTo
    end
end

@testitem "_OneTo tag — slice demotes to _UnitStep, axes() construction, eltype convert" begin
    using FastInterpolations: _to_float, _CachedRange, _AbstractUnitStep, _UnitStep, _OneTo
    tagof(x::_CachedRange) = typeof(x).parameters[3]

    o = _to_float(Base.OneTo(8), Float64)

    @testset "unit-step tag family + first() fold literal" begin
        @test _UnitStep <: _AbstractUnitStep
        @test _OneTo <: _AbstractUnitStep
        @test first(o) === 1.0                          # literal one(T), not the field
        @test first(_to_float(Base.OneTo(8), Float32)) === 1.0f0
    end

    @testset "slice demotes (lo≡1 invariant would be a lie on sub-ranges)" begin
        s = o[2:5]
        @test tagof(s) === _UnitStep           # never _OneTo, even when 1-based:
        @test s.lo === 2.0 && s.hi === 5.0 && length(s) == 4
        s1 = o[1:5]                            # a 1-based slice demotes too (type can't
        @test tagof(s1) === _UnitStep          # depend on the runtime slice start)
        @test s1.lo === 1.0
        @test tagof(view(o, 3:6)) === _UnitStep
        e = o[2:1]                             # empty slice
        @test tagof(e) === _UnitStep && length(e) == 0
    end

    @testset "eltype convert preserves _OneTo (lo ≡ 1 survives T conversion)" begin
        o32 = _to_float(o, Float32)
        @test o32 isa _CachedRange{Float32, Float32, _OneTo}
        @test o32.lo === 1.0f0
    end

    @testset "axes(data) construction — the idiomatic image route gets _OneTo" begin
        data = [0.1i + 0.3j for i in 1:8, j in 1:6]
        itp = linear_interp(axes(data), data)
        @test tagof(itp.grids[1]) === _OneTo
        @test tagof(itp.grids[2]) === _OneTo
        @test itp((3.5, 2.25)) === linear_interp((1:8, 1:6), data)((3.5, 2.25))
    end
end
