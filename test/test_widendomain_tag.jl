# Coverage for the `_WidenDomain` axis tag: the `_cached_range` construction
# factory, tag-dispatched `_domain_bounds`, and NoExtrap boundary acceptance.

@testitem "WidenDomain — _cached_range factory builds widened vs exact domain" begin
    using FastInterpolations: _cached_range, _WidenDomain, _Generic, _CachedRange

    w = _cached_range(_WidenDomain(), 0.0, 10.0, 1.0, 1.0, 11)
    @test w isa _CachedRange{Float64, Float64, _WidenDomain}
    @test w.lo == 0.0 && w.hi == 10.0                       # physical endpoints unchanged
    @test w.domain_lo == prevfloat(0.0)                     # widened bracket
    @test w.domain_hi == nextfloat(10.0)

    g = _cached_range(_Generic(), 0.0, 10.0, 1.0, 1.0, 11)
    @test g isa _CachedRange{Float64, Float64, _Generic}
    @test g.domain_lo == 0.0 && g.domain_hi == 10.0         # exact: fields equal lo/hi
end

@testitem "WidenDomain — _domain_bounds dispatches per tag" begin
    using FastInterpolations: _cached_range, _WidenDomain, _Generic, _domain_bounds, _to_float

    w = _cached_range(_WidenDomain(), 0.0, 10.0, 1.0, 1.0, 11)
    @test _domain_bounds(w) == (prevfloat(0.0), nextfloat(10.0))

    g = _cached_range(_Generic(), 0.0, 10.0, 1.0, 1.0, 11)
    @test _domain_bounds(g) == (0.0, 10.0)

    u = _to_float(1:11, Float64)                            # _UnitStep — exact domain
    @test _domain_bounds(u) == (first(u), last(u))
end

@testitem "WidenDomain — NoExtrap accepts the widened boundary, rejects beyond" begin
    using FastInterpolations: _cached_range, _WidenDomain, _Generic, _check_domain, NoExtrap

    w = _cached_range(_WidenDomain(), 0.0, 10.0, 1.0, 1.0, 11)
    @test _check_domain(w, 10.0, NoExtrap()) === nothing
    @test _check_domain(w, nextfloat(10.0), NoExtrap()) === nothing          # == domain_hi, in-domain
    @test_throws DomainError _check_domain(w, nextfloat(nextfloat(10.0)), NoExtrap())

    g = _cached_range(_Generic(), 0.0, 10.0, 1.0, 1.0, 11)
    @test _check_domain(g, 10.0, NoExtrap()) === nothing
    @test_throws DomainError _check_domain(g, nextfloat(10.0), NoExtrap())   # no cushion
end
