# N=1 scalar-query adapter (`_scalar_query` / `_ScalarQuery`).
#
# A 1-axis grid tuple collapses to the 1D engines, whose batch signature reads one
# scalar per index. An ND caller at N=1 may still hand over a POINT container (AoS
# `[(x,)]`, `Vector{Vector}`, shaped `Matrix{Tuple}`, `GriddedQuery`) that the ND
# side reads through the query protocol. `_scalar_query` is the single bridge:
#   - a numeric array or single-axis SoA is returned AS-IS (native hot path untouched)
#   - anything else becomes `_ScalarQuery`, a lazy `AbstractArray{T,M}` that reads
#     point k via `_query_extract` and projects it to its one coordinate.
# No copy, shape preserved, eltype from the protocol, point arity guarded.

@testitem "_scalar_query: numeric arrays and single-axis SoA pass through untouched" begin
    using FastInterpolations: _scalar_query

    qs = [0.1, 0.5, 0.9]
    @test _scalar_query(qs) === qs                       # bare Vector → identity
    qi = [1, 2, 3]
    @test _scalar_query(qi) === qi                       # eltype-agnostic identity
    M = reshape(collect(0.1:0.1:0.6), 2, 3)
    @test _scalar_query(M) === M                         # shaped bare → identity
    @test _scalar_query((qs,)) === qs                    # SoA → unwrap
    @test _scalar_query((M,)) === M                      # shaped SoA → unwrap
    r = 0.1:0.1:0.9
    @test _scalar_query(r) === r                         # ranges are numeric arrays
end

@testitem "_scalar_query: point containers become a lazy scalar view" begin
    using FastInterpolations: _scalar_query, _ScalarQuery

    qs = [0.1, 0.5, 0.9]
    aos = [(q,) for q in qs]
    v = _scalar_query(aos)
    @test v isa _ScalarQuery
    @test v isa AbstractVector{Float64}
    @test eltype(v) === Float64
    @test size(v) == (3,)
    @test length(v) == 3
    @test Base.IndexStyle(typeof(v)) === IndexLinear()
    @test v[2] == 0.5
    @test collect(v) == qs
    # lazy: no copy — the view reflects the parent
    aos[1] = (0.3,)
    @test v[1] == 0.3

    # Vector{Vector}: dynamic-length points
    vov = [[q] for q in qs]
    @test eltype(_scalar_query(vov)) === Float64
    @test collect(_scalar_query(vov)) == qs

    # shaped AoS: shape preserved, column-major linear order
    Mt = reshape([(q,) for q in 0.1:0.1:0.6], 2, 3)
    vm = _scalar_query(Mt)
    @test vm isa AbstractMatrix{Float64}
    @test size(vm) == (2, 3)
    @test vm[2, 3] == 0.6
    @test vm[6] == 0.6
    @test collect(vm) == reshape(collect(0.1:0.1:0.6), 2, 3)
    @test collect(eachindex(vm, zeros(2, 3))) == collect(1:6)   # engines iterate `eachindex(q, out)`

    # GriddedQuery at N=1 is a 1-axis point container
    gq = GriddedQuery((qs,))
    @test collect(_scalar_query(gq)) == qs
    @test size(_scalar_query(gq)) == (3,)

    # scalar eltype is taken from the point type
    @test eltype(_scalar_query([(1.0f0,)])) === Float32
    @test eltype(_scalar_query([(1,)])) === Int

    # empty batch: shape and type without probing a point
    e = _scalar_query(Tuple{Float64}[])
    @test size(e) == (0,)
    @test eltype(e) === Float64
    @test collect(e) == Float64[]
end

@testitem "_scalar_query: point arity is guarded at every access" begin
    using FastInterpolations: _scalar_query

    # static over-dimensioned points (2-D points on a 1-axis grid): reading throws
    two = [(0.1, 0.2), (0.3, 0.4)]
    @test_throws DimensionMismatch _scalar_query(two)[1]
    # dynamic ragged: point 1 is fine, point 2 is malformed → caught at ITS access
    # (the ND `_query_check_ndims` probe inspects only point 1)
    ragged = [[0.1], [0.2, 0.3]]
    r = _scalar_query(ragged)
    @test r[1] == 0.1
    @test_throws DimensionMismatch r[2]
    @test_throws DimensionMismatch _scalar_query([Float64[]])[1]    # empty point
    # the view is a proper AbstractArray: outer bounds are checked (a `@boundscheck`
    # block, so it is gone — like every other one — under `--check-bounds=no`)
    if Base.JLOptions().check_bounds != 2
        @test_throws BoundsError _scalar_query([(0.1,)])[2]
        @test_throws BoundsError _scalar_query([(0.1,)])[0]
    end
end

@testitem "_scalar_query: the 1D engines consume the view like a bare vector" setup = [AllocConstants] begin
    using FastInterpolations: _scalar_query
    using Test: @inferred

    x = collect(range(0.0, 1.0, length = 40))
    y = @. sin(7x)
    qs = collect(range(0.02, 0.98, length = 25))
    aos = [(q,) for q in qs]
    out = similar(qs)
    ref = similar(qs)

    @noinline run!(f!::F, o, g, d, q) where {F} = (f!(o, g, d, q); nothing)

    for f! in (
            constant_interp!, linear_interp!, quadratic_interp!, cubic_interp!,
            pchip_interp!, cardinal_interp!, akima_interp!,
        )
        v = _scalar_query(aos)
        f!(ref, x, y, qs)
        f!(out, x, y, v)
        @test out == ref                                    # same engine, same reads → bit-identical
        run!(f!, out, x, y, v); run!(f!, out, x, y, v)      # warm
        @test (@allocated run!(f!, out, x, y, v)) <= ALLOC_THRESHOLD
        @test (@inferred f!(out, x, y, v)) === out
    end

    # allocating form follows the view's shape (Matrix{Tuple} → Matrix)
    Mt = reshape(aos[1:24], 4, 6)
    r = cubic_interp(x, y, _scalar_query(Mt))
    @test size(r) == (4, 6)
    @test vec(r) == cubic_interp(x, y, qs[1:24])

    # per-call kwargs are the engine's own: deriv / extrap / hint flow unchanged
    @test cubic_interp(x, y, _scalar_query(aos); deriv = DerivOp(1)) == cubic_interp(x, y, qs; deriv = DerivOp(1))
    oob = [(-0.5,), (1.5,)]
    @test linear_interp(x, y, _scalar_query(oob); extrap = ClampExtrap()) == linear_interp(x, y, [-0.5, 1.5]; extrap = ClampExtrap())
    # NoExtrap OOB: the view gets exactly the bare batch's outcome (a DomainError under
    # `--check-bounds=yes`; the batch domain gate is elided with checks off, for both)
    outcome(q) = try
        (:value, linear_interp(x, y, q))
    catch e
        (:error, typeof(e))
    end
    @test outcome(_scalar_query(oob)) == outcome([-0.5, 1.5])
    Base.JLOptions().check_bounds != 2 && @test outcome([-0.5, 1.5]) == (:error, DomainError)
    h1 = Ref(1); h2 = Ref(1)
    linear_interp(x, y, qs; hint = h1)
    linear_interp(x, y, _scalar_query(aos); hint = h2)
    @test h1[] == h2[] > 1
end
