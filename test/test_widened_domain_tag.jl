# Coverage for the `_WidenedDomain` axis tag: the `_cached_range` construction
# factory, tag-dispatched `_domain_bounds`, NoExtrap boundary acceptance, the
# slice/convert re-widen contract, the cubic-cache type invariant, and the
# zero-alloc persistent NoExtrap scalar hot path.

@testitem "WidenedDomain — _cached_range factory builds widened vs exact domain" begin
    using FastInterpolations: _cached_range, _WidenedDomain, _Generic, _CachedRange

    w = _cached_range(_WidenedDomain(), 0.0, 10.0, 1.0, 1.0, 11)
    @test w isa _CachedRange{Float64, Float64, _WidenedDomain}
    @test w.lo == 0.0 && w.hi == 10.0                       # physical endpoints unchanged
    @test w.domain_lo == prevfloat(0.0)                     # widened bracket
    @test w.domain_hi == nextfloat(10.0)

    g = _cached_range(_Generic(), 0.0, 10.0, 1.0, 1.0, 11)
    @test g isa _CachedRange{Float64, Float64, _Generic}
    @test g.domain_lo == 0.0 && g.domain_hi == 10.0         # exact: fields equal lo/hi
end

@testitem "WidenedDomain — _domain_bounds dispatches per tag" begin
    using FastInterpolations: _cached_range, _WidenedDomain, _Generic, _domain_bounds, _to_float

    w = _cached_range(_WidenedDomain(), 0.0, 10.0, 1.0, 1.0, 11)
    @test _domain_bounds(w) == (prevfloat(0.0), nextfloat(10.0))

    g = _cached_range(_Generic(), 0.0, 10.0, 1.0, 1.0, 11)
    @test _domain_bounds(g) == (0.0, 10.0)

    u = _to_float(1:11, Float64)                            # _UnitStep — exact domain
    @test _domain_bounds(u) == (first(u), last(u))
end

@testitem "WidenedDomain — NoExtrap accepts the widened boundary, rejects beyond" begin
    using FastInterpolations: _cached_range, _WidenedDomain, _Generic, _check_domain, NoExtrap

    w = _cached_range(_WidenedDomain(), 0.0, 10.0, 1.0, 1.0, 11)
    @test _check_domain(w, 10.0, NoExtrap()) === nothing
    @test _check_domain(w, nextfloat(10.0), NoExtrap()) === nothing          # == domain_hi, in-domain
    @test_throws DomainError _check_domain(w, nextfloat(nextfloat(10.0)), NoExtrap())

    g = _cached_range(_Generic(), 0.0, 10.0, 1.0, 1.0, 11)
    @test _check_domain(g, 10.0, NoExtrap()) === nothing
    @test_throws DomainError _check_domain(g, nextfloat(10.0), NoExtrap())   # no cushion
end

@testitem "WidenedDomain — slice and type-convert re-derive the domain bracket" begin
    using FastInterpolations: _cached_range, _WidenedDomain, _Generic, _UnitStep,
        _CachedRange, _domain_bounds, _to_float

    w = _cached_range(_WidenedDomain(), 0.0, 10.0, 1.0, 1.0, 11)

    # Interior slice: tag preserved, bracket re-widened on the new endpoints.
    s = w[3:8]
    @test s isa _CachedRange{Float64, Float64, _WidenedDomain}
    @test _domain_bounds(s) == (prevfloat(s.lo), nextfloat(s.hi))

    # Boundary-touching slice recovers the ORIGINAL left cushion (new_lo === w.lo).
    sb = w[1:6]
    @test sb.lo == w.lo
    @test _domain_bounds(sb)[1] == prevfloat(w.lo) == w.domain_lo

    # Exact tags stay exact through a slice; _UnitStep stays unit-step.
    gs = _cached_range(_Generic(), 0.0, 10.0, 1.0, 1.0, 11)[3:8]
    @test gs isa _CachedRange{Float64, Float64, _Generic}
    @test _domain_bounds(gs) == (gs.lo, gs.hi)
    us = _to_float(1:11, Float64)[3:8]
    @test us isa _CachedRange{Float64, Float64, _UnitStep}
    @test _domain_bounds(us) == (us.lo, us.hi)

    # Type-convert of a _WidenedDomain source stays widened (cushion recomputed on T).
    wc = _to_float(_cached_range(_WidenedDomain(), 0.0f0, 10.0f0, 1.0f0, 1.0f0, 11), Float64)
    @test wc isa _CachedRange{Float64, Float64, _WidenedDomain}
    @test _domain_bounds(wc) == (prevfloat(Float64(0.0f0)), nextfloat(Float64(10.0f0)))
end

@testitem "WidenedDomain — _cached_axis_type matches _to_float output type (cubic-cache invariant)" begin
    using FastInterpolations: _cached_range, _WidenedDomain, _Generic, _to_float, _cached_axis_type

    # The cubic cache pool predicts the banked grid type via `_cached_axis_type`; it
    # must equal the concrete type `_to_float` actually produces — for every tag and
    # both the identity (same-T) and convert (diff-T) paths. Pins the re-widen
    # tag-preservation coupling on any arch (a real _WidenedDomain grid only arises on
    # x86, but the type-level invariant is arch-independent).
    srcs = (
        _cached_range(_WidenedDomain(), 0.0f0, 10.0f0, 1.0f0, 1.0f0, 11),
        _cached_range(_Generic(), 0.0f0, 10.0f0, 1.0f0, 1.0f0, 11),
        _to_float(1:11, Float32),   # _UnitStep
    )
    for x in srcs
        @test _cached_axis_type(typeof(x), Float64) === typeof(_to_float(x, Float64))  # convert
        @test _cached_axis_type(typeof(x), Float32) === typeof(_to_float(x, Float32))  # identity
    end
end

@testitem "WidenedDomain — x86_64 StepRangeLen fast path produces a real _WidenedDomain (arch-gated)" begin
    using FastInterpolations: _to_float, _CachedRange, _WidenedDomain

    w = _to_float(0.0:0.1:1.0, Float64)
    if Sys.ARCH === :x86_64
        # The TwicePrecision fast path tags the grid _WidenedDomain with a genuine
        # ±1-ULP bracket; this guards against a future "simplify the factory" that
        # would silently drop the cushion on the only arch that needs it.
        @test w isa _CachedRange{Float64, Float64, _WidenedDomain}
        @test w.domain_lo === prevfloat(w.lo)
        @test w.domain_hi === nextfloat(w.hi)
    else
        # On aarch64 the fast path is @static-compiled out: the same grid is exact.
        @test !(w isa _CachedRange{Float64, Float64, _WidenedDomain})
        @test w.domain_lo == w.lo && w.domain_hi == w.hi
    end
end

@testitem "WidenedDomain — persistent NoExtrap scalar eval is zero-alloc" setup = [AllocConstants] begin
    using FastInterpolations: linear_interp

    # The PR routes the NoExtrap _CachedRange domain check through `_domain_bounds`;
    # pin that the persistent scalar hot path stays allocation-free. A function
    # barrier keeps the measured call type-stable; ALLOC_THRESHOLD is 0 on Julia
    # ≥ 1.12, with a small LTS slack.
    function _alloc_noextrap(g, q)
        y = collect(Float64, range(0.0, 1.0; length = length(g)))
        itp = linear_interp(g, y)                          # default extrap = NoExtrap()
        itp(q); itp(q)                                     # warm up
        return @allocated itp(q)
    end
    @test _alloc_noextrap(0.0:0.1:1.0, 0.55) <= ALLOC_THRESHOLD   # _Generic / _WidenedDomain (x86)
    @test _alloc_noextrap(1:11, 5.5) <= ALLOC_THRESHOLD          # _UnitStep
end
