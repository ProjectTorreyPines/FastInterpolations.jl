# GridIdx in BATCH queries.
#
# `GridIdx(k)` names grid node k. The scalar 1D one-shot resolves it at the door and the
# ND batch loop resolves it per point, but the 1D BATCH door used to read the raw batch
# (`minimum`/`maximum`, the ordering probe) and two `GridIdx` of one type have no `<`, so
# every bare `Vector{GridIdx}` batch failed — and, once the N=1 collapse routed every
# container to the 1D engine, so did an AoS `[(GridIdx(k),)]`. The door now validates a
# GridIdx batch by index (`_check_domain` GridIdx arms) and skips the ordering probe.
# Contract pinned here: `GridIdx(k)` IS the coordinate `x[k]` — a GridIdx batch equals
# the coordinate batch `x[ks]` (values and buffer eltype) on every route and under every
# extrapolation mode; an index outside `1:length(x)` is rejected with the scalar
# resolve's `ArgumentError` under every mode, before anything is read.

@testitem "GridIdx batch on the 1D door: every entry names a node" begin
    using FastInterpolations

    # first / interior / last nodes, then an 8-point batch (long enough for the
    # AutoSearch ordering probe) that is descending and repeats a node
    ks = [1, 3, 9, 12]
    g = [GridIdx(k) for k in ks]
    ks8 = [12, 9, 9, 3, 1, 5, 7, 2]
    g8 = [GridIdx(k) for k in ks8]
    # every mode the door can be called with: a node is never out of domain, and an
    # invalid index is rejected under each of them
    modes = (ClampExtrap(), FillExtrap(NaN), WrapExtrap(), ExtendExtrap(), FastInterpolations.InBounds())
    # Vector, general range, unit-step float range and Int unit range: the door has a
    # `_check_domain` arm per grid kind (AbstractVector / _CachedRange / unit-step), so
    # every grid kind must reach its own twin
    grids = (collect(range(0.0, 1.0, length = 12)), range(0.0, 1.0, length = 12), 1.0:12.0, 1:12)

    for x in grids, (f, f!) in (
                (constant_interp, constant_interp!), (linear_interp, linear_interp!),
                (quadratic_interp, quadratic_interp!), (cubic_interp, cubic_interp!),
                (pchip_interp, pchip_interp!), (akima_interp, akima_interp!),
                (cardinal_interp, cardinal_interp!),
            )
        y = @. sin(4x) + 0.2x
        # scalar form: GridIdx(k) ≡ the coordinate x[k] (first / interior / last node)
        @test f(x, y, GridIdx(1)) == f(x, y, x[1])
        @test f(x, y, GridIdx(3)) == f(x, y, x[3])
        @test f(x, y, GridIdx(12)) == f(x, y, x[12])
        ref = f(x, y, x[ks])
        ref8 = f(x, y, x[ks8])
        @test f(x, y, g) == ref                                    # allocating batch
        @test eltype(f(x, y, g)) === eltype(ref)                   # same buffer type, too
        @test f(x, y, g8) == ref8                                  # long, unsorted, repeated
        out = zeros(length(ks))
        @test f!(out, x, y, g) === out                             # in-place batch
        @test out == ref
        # derivative at a node: same cell, same kernel as the coordinate batch
        @test f(x, y, g; deriv = DerivOp(1)) == f(x, y, x[ks]; deriv = DerivOp(1))
        # a `hint` is accepted
        h = Ref(1)
        @test f(x, y, g; hint = h) == ref
        for e in modes
            @test f(x, y, g; extrap = e) == ref
            @test f(x, y, g8; extrap = e) == ref8
        end
        # empty batch
        @test f(x, y, GridIdx{Float64}[]) == Float64[]
        # an index outside 1:length(x) is rejected with the scalar resolve's error, under
        # every extrapolation mode — an index has no extrapolation meaning
        @test_throws ArgumentError f(x, y, [GridIdx(13)])
        @test_throws ArgumentError f(x, y, [GridIdx(3), GridIdx(14)])
        for e in modes
            @test_throws ArgumentError f(x, y, [GridIdx(13)]; extrap = e)
        end
        @test_throws ArgumentError f!(out, x, y, [GridIdx(13), GridIdx(1), GridIdx(2), GridIdx(3)])
    end
end

@testitem "GridIdx batch through the N=1 collapse: bare, SoA, AoS, shaped, unified" begin
    using FastInterpolations

    ks = [1, 3, 9, 12]
    g = [GridIdx(k) for k in ks]
    aos = [(GridIdx(k),) for k in ks]
    ks8 = [12, 9, 9, 3, 1, 5, 7, 2]
    aos8 = [(GridIdx(k),) for k in ks8]
    # Vector, general range, unit-step float range and Int unit range: the door has a
    # `_check_domain` arm per grid kind (AbstractVector / _CachedRange / unit-step), so
    # every grid kind must reach its own twin
    grids = (collect(range(0.0, 1.0, length = 12)), range(0.0, 1.0, length = 12), 1.0:12.0, 1:12)

    for x in grids, (m, f, f!) in (
                (ConstantInterp(), constant_interp, constant_interp!), (LinearInterp(), linear_interp, linear_interp!),
                (QuadraticInterp(), quadratic_interp, quadratic_interp!), (CubicInterp(), cubic_interp, cubic_interp!),
                (PchipInterp(), pchip_interp, pchip_interp!), (AkimaInterp(), akima_interp, akima_interp!),
                (CardinalInterp(), cardinal_interp, cardinal_interp!),
            )
        y = @. sin(4x) + 0.2x
        ref = f(x, y, x[ks])
        @test f((x,), y, g) == ref                       # bare vector on a 1-tuple grid
        @test f((x,), y, (g,)) == ref                    # SoA
        @test f((x,), y, aos) == ref                     # AoS — evaluated on the ND route before
        @test eltype(f((x,), y, aos)) === eltype(ref)
        @test f((x,), y, aos8) == f(x, y, x[ks8])
        out = zeros(length(ks))
        f!(out, (x,), y, aos)
        @test out == ref
        @test f((x,), y, (GridIdx(3),)) == f((x,), y, (x[3],))   # scalar tuple form
        @test interp((x,), y, aos; method = m) == ref    # unified API
        @test interp((x,), y, g; method = m) == ref
        @test_throws ArgumentError f((x,), y, [(GridIdx(13),)])
    end

    # shaped AoS keeps its shape (Vector and unit-step range grids)
    M = reshape([(GridIdx(k),) for k in (2, 5, 7, 11)], 2, 2)
    E = reshape(Tuple{GridIdx{Float64}}[], 0, 2)          # shaped empty adapter
    for x in (collect(range(0.0, 1.0, length = 12)), 1.0:12.0)
        y = @. sin(4x) + 0.2x
        Mref = cubic_interp((x,), y, reshape(x[[2, 5, 7, 11]], 2, 2))
        @test cubic_interp((x,), y, M) == Mref
        for e in (ClampExtrap(), FillExtrap(NaN), WrapExtrap(), ExtendExtrap())
            @test cubic_interp((x,), y, M; extrap = e) == Mref
        end
        outM = zeros(2, 2)
        @test interp!(outM, (x,), y, M; method = CubicInterp()) === outM   # unified in-place, shaped
        @test outM == Mref
        @test size(cubic_interp((x,), y, E)) == (0, 2)
    end
end

@testitem "GridIdx batch on a persistent 1D interpolant" begin
    using FastInterpolations

    ks = [1, 3, 9, 12]
    g = [GridIdx(k) for k in ks]
    # Vector, general range, unit-step float range and Int unit range: the door has a
    # `_check_domain` arm per grid kind (AbstractVector / _CachedRange / unit-step), so
    # every grid kind must reach its own twin
    grids = (collect(range(0.0, 1.0, length = 12)), range(0.0, 1.0, length = 12), 1.0:12.0, 1:12)

    for x in grids, f in (
                constant_interp, linear_interp, quadratic_interp, cubic_interp,
                pchip_interp, akima_interp, cardinal_interp,
            )
        y = @. sin(4x) + 0.2x
        itp = f(x, y)
        ref = itp(x[ks])
        @test itp(GridIdx(1)) == itp(x[1])               # scalar form, first / last node
        @test itp(GridIdx(12)) == itp(x[12])
        @test itp(g) == ref                              # batch
        @test eltype(itp(g)) === eltype(ref)
        out = zeros(length(ks))
        @test itp(out, g) === out
        @test out == ref
        @test itp((g,)) == ref                           # ND-style single-axis SoA
        @test_throws ArgumentError itp([GridIdx(13)])
        itp_t = f((x,), y)                               # collapsed constructor, same object type
        @test itp_t(g) == ref
    end
end

# `Base.convert(::Type{T}, g::GridIdx) where {T <: Number}` also matches T = GridIdx{T}
# itself; on Julia 1.10 that arm wins over Base's identity `convert(::Type{T}, ::T)` and
# recurses into `GridIdx{Float64}(::Float64)`, so even the vector literal above cannot
# be built there; an explicit `convert` fails on every version. Self-conversion must be
# the identity.
@testitem "GridIdx self-conversion is the identity (vector literals build everywhere)" begin
    using FastInterpolations

    g3 = GridIdx(3)
    @test convert(GridIdx{Float64}, g3) === g3
    @test [GridIdx(3), GridIdx(9)] isa Vector{GridIdx{Float64}}
    v = Vector{GridIdx{Float64}}(undef, 1)
    v[1] = g3
    @test v[1] === g3
    @test map(GridIdx, [3, 9]) == [GridIdx(3), GridIdx(9)]
end

# Output eltype: a GridIdx query resolves to the node coordinate, so every allocating
# entry must size its buffer from the axis type, not from the unresolved wrapper's
# `Float64` payload — else constant selects Int data into a Float64 buffer and rounds
# above 2^53 while the coordinate batch keeps the exact Int.
@testitem "GridIdx batch on an Int grid: constant keeps Int data exact on every allocating entry" begin
    using FastInterpolations

    y = fill(2^53 + 1, 12)                       # not representable in Float64
    ks = [1, 3, 12]
    g = GridIdx.(ks)
    aos = [(GridIdx(k),) for k in ks]
    for x in (1:12, collect(1:12))
        ref = constant_interp(x, y, x[ks])
        @test ref == y[ks]
        @test eltype(ref) === Int
        r = constant_interp(x, y, g)                          # 1D one-shot
        @test r == ref
        @test eltype(r) === Int
        itp = constant_interp(x, y)                           # persistent
        @test itp(g) == ref
        @test eltype(itp(g)) === Int
        @test constant_interp((x,), y, aos) == ref            # N=1 collapse: AoS view and bare
        @test constant_interp((x,), y, g) == ref
        u = interp((x,), y, aos; method = ConstantInterp())   # unified
        @test u == ref
        @test eltype(u) === Int
    end
    # the ND route (N = 2) states the same coordinate equivalence (its constant kernel
    # rounds Int data through a Float64 `one(dL)` for coordinate queries as well — a
    # separate, pre-existing ND matter; only the GridIdx ↔ coordinate parity is pinned)
    X = (1:12, 1:5)
    D = fill(2^53 + 1, 12, 5)
    pts = [(GridIdx(3), GridIdx(2)), (GridIdx(12), GridIdx(5))]
    refnd = constant_interp(X, D, [(3, 2), (12, 5)])
    @test constant_interp(X, D, pts) == refnd
    @test eltype(constant_interp(X, D, pts)) === eltype(refnd)
end

# The GridIdx door costs nothing past the output: the validation pass and the per-point
# short-circuit allocate nothing, the N=1 AoS view is a stack struct over the caller's
# array, and the result type is concretely inferred. Measured through `@noinline`
# barriers with every argument prebuilt (a tuple built inside `@allocated` is boxed by
# the caller, on the bare route just the same).
@testitem "GridIdx batch door is allocation-free and inferred" setup = [AllocConstants] begin
    using FastInterpolations
    using Test: @inferred

    ks = [12, 9, 9, 3, 1, 5, 7, 2, 11, 4, 6, 8]
    g = GridIdx.(ks)
    aos = [(GridIdx(k),) for k in ks]
    out = zeros(length(ks))
    @noinline run!(f!::F, o, x, d, q) where {F} = (f!(o, x, d, q); nothing)
    @noinline irun!(itp::I, o, q) where {I} = (itp(o, q); nothing)
    @noinline urun!(o, gx, d, q, m::M) where {M} = (interp!(o, gx, d, q; method = m); nothing)

    for x in (collect(range(0.0, 1.0, length = 12)), range(0.0, 1.0, length = 12), 1.0:12.0)
        y = @. sin(4x) + 0.2x
        gx = (x,)
        for (m, f, f!) in (
                (ConstantInterp(), constant_interp, constant_interp!), (LinearInterp(), linear_interp, linear_interp!),
                (QuadraticInterp(), quadratic_interp, quadratic_interp!), (CubicInterp(), cubic_interp, cubic_interp!),
                (PchipInterp(), pchip_interp, pchip_interp!), (AkimaInterp(), akima_interp, akima_interp!),
                (CardinalInterp(), cardinal_interp, cardinal_interp!),
            )
            itp = f(x, y)
            for _ in 1:2                                                     # warm every form
                run!(f!, out, x, y, g); run!(f!, out, gx, y, aos); irun!(itp, out, g); urun!(out, gx, y, aos, m)
            end
            @test (@allocated run!(f!, out, x, y, g)) <= ALLOC_THRESHOLD          # bare batch
            @test (@allocated run!(f!, out, gx, y, aos)) <= ALLOC_THRESHOLD       # N=1 AoS view
            @test (@allocated irun!(itp, out, g)) <= ALLOC_THRESHOLD              # persistent
            @test (@allocated urun!(out, gx, y, aos, m)) <= ALLOC_THRESHOLD       # unified
            @test (@inferred f!(out, x, y, g)) === out
            @test (@inferred f(x, y, g)) isa Vector{Float64}                       # output is the only allocation
            @test (@inferred itp(g)) isa Vector{Float64}
        end
    end
end

# Series (K×Q) never resolves GridIdx: the anchor builder hands the raw token to
# `_anchor_loc` and the coordinate arithmetic consumes its NaN payload. A separate
# surface from the 1D batch door — recorded here as the follow-up contract.
@testitem "GridIdx on the Series surface (follow-up: unresolved today)" begin
    using FastInterpolations

    x = collect(range(0.0, 1.0, length = 12))
    y = @. sin(4x) + 0.2x
    Y = hcat(y, 2 .* y)

    for f in (constant_interp, linear_interp, quadratic_interp, cubic_interp)
        @test_broken f(x, Series(Y), [GridIdx(3)]) == f(x, Series(Y), [x[3]])
        sitp = f(x, Series(Y))
        @test_broken sitp(GridIdx(3)) == sitp(x[3])
        @test_broken sitp([GridIdx(3)]) == sitp([x[3]])
    end
end
