# Behavior-pinning tests for N=1 tuple-grid collapse (`axes(y)` convenience).
#
# Contract being pinned:
#   1. A 1-length grid tuple `foo_interp((x,), y)` COLLAPSES to the genuine 1D
#      interpolant (`<: AbstractInterpolant1D`), not an N=1 `*InterpolantND`.
#      This is what makes `foo_interp(axes(y), y)` take the fast 1D path.
#   2. The collapsed interpolant is VALUE-IDENTICAL to `foo_interp(x, y)` — the
#      grid tuple is merely unwrapped, nothing about the math changes.
#   3. Per-axis kwargs given ND-style as 1-tuples (`extrap=(WrapExtrap(),)`) are
#      unwrapped to their scalar form for the 1D constructor.
#   4. 1D interpolants accept ND-style tuple queries so collapse is transparent
#      to generic tensor code: `itp((x,))`, `itp(out,(xv,))`, `itp((xv,))`, and
#      per-axis kwargs `itp((x,); extrap=(WrapExtrap(),))` / `deriv=(op,)`.
#   5. One-shot batch on a 1-tuple grid reaches the 1D engine for EVERY batch
#      container — bare vector, SoA `(xv,)`, AoS `[(x,)]`, `Vector{Vector}`, shaped
#      AoS, `GriddedQuery` — via `_scalar_query` (issue #204). Only cubic/quadratic
#      explicit `coeffs = OnTheFly()` stays on the ND internals.

@testitem "N=1 tuple-grid collapses to 1D" begin
    using FastInterpolations
    using FastInterpolations: AbstractInterpolant1D, AbstractInterpolantND

    x = collect(1.0:10.0)
    y = @. sin(x) + 0.3 * x
    xq = 3.7

    # Every method whose 2-arg `(grid, data)` form has an ND path.
    for f in (
            linear_interp, cubic_interp, quadratic_interp, constant_interp,
            pchip_interp, akima_interp, cardinal_interp,
        )
        itp_nd = f((x,), y)          # tuple grid
        itp_1d = f(x, y)             # genuine 1D
        @test itp_nd isa AbstractInterpolant1D
        @test !(itp_nd isa AbstractInterpolantND)
        # value-identical to the 1D path (scalar query)
        @test itp_nd(xq) == itp_1d(xq)
        # `axes(y)` — the motivating OneTo case
        itp_ax = f(axes(y), y)
        @test itp_ax isa AbstractInterpolant1D
    end
end

@testitem "N=1 collapse is type-stable and yields the 1D type" begin
    using FastInterpolations
    using FastInterpolations: AbstractInterpolant1D, AbstractInterpolantND
    using Test: @inferred

    x = collect(1.0:10.0)
    y = @. sin(x) + 0.3 * x

    # The collapse forwarder must not introduce any inference instability beyond
    # the genuine 1D constructor: `@inferred` succeeds AND the returned object is a
    # 1D interpolant. `@inferred` throws if the call site is not concretely inferred,
    # so pairing it with the `isa` check pins both stability and the collapsed type.
    for f in (
            linear_interp, cubic_interp, quadratic_interp, constant_interp,
            pchip_interp, akima_interp, cardinal_interp,
        )
        itp = @inferred f((x,), y)
        @test itp isa AbstractInterpolant1D
        @test !(itp isa AbstractInterpolantND)
        # collapse produces the SAME concrete type as the direct 1D call
        @test typeof(itp) === typeof(f(x, y))
    end

    # Hot path: value queries through the 1D tuple-query shims stay concretely typed.
    itp = linear_interp(x, y)
    @test (@inferred itp((3.7,))) isa Float64
    xv = [2.3, 5.1, 8.8]; out = similar(xv)
    @test (@inferred itp(out, (xv,))) isa Vector{Float64}
    @test (@inferred itp((xv,))) isa Vector{Float64}
end

@testitem "2D path unchanged by N=1 collapse (no regression)" begin
    using FastInterpolations
    using FastInterpolations: AbstractInterpolantND
    using Test: @inferred

    x = collect(1.0:6.0)
    z = collect(1.0:5.0)
    data = [sin(xi) * cos(zj) for xi in x, zj in z]
    q = (3.3, 2.7)

    # The more-specific `Tuple{AbstractVector}` collapse method must NOT shadow the
    # generic `NTuple{N}` ND constructor for N≥2 — a 2-tuple grid still builds ND.
    for f in (
            linear_interp, cubic_interp, quadratic_interp, constant_interp,
            pchip_interp, akima_interp, cardinal_interp,
        )
        itp = @inferred f((x, z), data)
        @test itp isa AbstractInterpolantND
        @test itp(q) isa Real            # ND tuple query still works
    end
end

@testitem "N=1 collapse unwraps per-axis kwargs" begin
    using FastInterpolations

    x = collect(1.0:10.0)
    y = @. cos(x)
    xq = 6.3

    # ND-style 1-tuple extrap must unwrap to the scalar 1D form.
    itp_t = linear_interp((x,), y; extrap = (WrapExtrap(),))
    itp_s = linear_interp(x, y; extrap = WrapExtrap())
    @test itp_t(11.5) == itp_s(11.5)   # exercises the wrap branch (OOB)
    @test itp_t(xq) == itp_s(xq)
end

@testitem "cubic N=1 collapse: PreCompute → 1D, OnTheFly stays ND" begin
    using FastInterpolations
    using FastInterpolations: AbstractInterpolant1D, AbstractInterpolantND

    x = collect(1.0:10.0)
    y = @. sin(x)
    q = 4.6

    # 1D cubic has no `coeffs` (it is inherently PreCompute) — the default/PreCompute
    # collapse must reach the lean 1D path without leaking the ND-only kwarg.
    itp_pc = cubic_interp((x,), y)
    @test itp_pc isa AbstractInterpolant1D
    @test cubic_interp((x,), y; coeffs = PreCompute()) isa AbstractInterpolant1D
    @test itp_pc(q) == cubic_interp(x, y)(q)

    # OnTheFly has no 1D equivalent → the N=1 ND (Hetero) path is preserved.
    itp_otf = cubic_interp((x,), y; coeffs = OnTheFly())
    @test itp_otf isa AbstractInterpolantND
    @test itp_otf((q,)) isa Real

    # The unified `interp` N=1 path (routes through the method fn) still evaluates.
    @test interp((x,), reshape(y, :); method = (CubicInterp(),))((q,)) ≈ cubic_interp(x, y)(q) rtol = 1.0e-12
end

@testitem "N=1 scalar one-shot collapses to a scalar (not a 1-elem Vector)" begin
    using FastInterpolations
    using FastInterpolations: AbstractInterpolantND

    x = collect(1.0:10.0)
    y = @. sin(x) + 0.3 * x
    q = 4.6

    for f in (
            linear_interp, cubic_interp, quadratic_interp, constant_interp,
            pchip_interp, akima_interp, cardinal_interp,
        )
        # Bare scalar one-shot on a 1-tuple grid must return a SCALAR — not the ND
        # `[val]` length-1 batch. A bare `q` is sugar for the scalar query `(q,)`, so
        # it routes to the ND scalar one-shot; `(q,)` already returned a scalar.
        v_bare = f((x,), y, q)
        @test v_bare isa Real
        @test v_bare == f((x,), y, (q,))       # bare ≡ tuple scalar query
        @test v_bare ≈ f(x, y, q) rtol = 1.0e-12  # matches the 1D one-shot (within ULP)
        # OOB bare scalar still errors (NoExtrap), not silently returning a vector.
        @test_throws Exception f((x,), y, -5.0)
    end

    # Batch one-shot is unchanged — a plain vector query returns a Vector.
    xq = [2.5, 7.5]
    @test linear_interp((x,), y, xq) isa Vector
    @test linear_interp((x,), y, xq) == linear_interp(x, y, xq)
end

@testitem "N=1 batch one-shot collapses to the 1D one-shot" begin
    using FastInterpolations

    x = collect(1.0:20.0)
    y = @. sin(x) + 0.3 * x
    xq = collect(range(2.0, 19.0, length = 40))
    out = similar(xq)

    for f in (
            linear_interp, cubic_interp, quadratic_interp, constant_interp,
            pchip_interp, akima_interp, cardinal_interp,
        )
        # A 1-tuple grid batch one-shot forwards to the 1D one-shot → bit-identical
        # (it *is* the 1D path), returning a Vector.
        v = f((x,), y, xq)
        @test v isa Vector
        @test v == f(x, y, xq)
    end

    # In-place batch collapses too.
    linear_interp!(out, (x,), y, xq)
    ref = similar(xq); linear_interp!(ref, x, y, xq)
    @test out == ref

    # SoA `(xv,)` on a 1-tuple grid also collapses to the 1D batch (single-axis SoA
    # is logically the same vectorized domain check as a bare 1D vector).
    for f in (
            linear_interp, cubic_interp, quadratic_interp, constant_interp,
            pchip_interp, akima_interp, cardinal_interp,
        )
        @test f((x,), y, (xq,)) == f(x, y, xq)
    end
    linear_interp!(out, (x,), y, (xq,))
    @test out == ref
end

@testitem "N=1 batch collapse tolerates the ND-only `hint` kwarg (GridIdx regression)" begin
    using FastInterpolations

    x = collect(1.0:20.0)
    y = @. sin(x) + 0.3 * x
    xq = collect(range(2.0, 19.0, length = 6))
    out = similar(xq)
    ref = similar(xq)

    # The ND batch one-shot threads a per-axis `hint` (mutable search state that
    # advances through the batch). The collapse forwards it to the 1D one-shot, which
    # now accepts `hint` (previously only the scalar one-shot / persistent callable did).
    # `hint = nothing` (the GridIdx / NoInterp pre-slice default) must not throw; a real
    # `(h,)` must advance `h[]` to the last query's cell — matching the ND N=1 contract.
    for f in (
            linear_interp, cubic_interp, quadratic_interp, constant_interp,
            pchip_interp, akima_interp, cardinal_interp,
        )
        @test f((x,), y, xq; hint = nothing) == f(x, y, xq)      # allocating, bare vector
        @test f((x,), y, (xq,); hint = nothing) == f(x, y, xq)   # allocating, SoA
        # A real hint advances to the same cell the direct 1D one-shot lands on.
        hc = Ref(1); f((x,), y, xq; hint = (hc,))
        h1 = Ref(1); f(x, y, xq; hint = h1)
        @test hc[] == h1[] > 1
        hs = Ref(1); f((x,), y, (xq,); hint = (hs,)); @test hs[] == h1[]   # SoA advances too
    end
    for f! in (
            linear_interp!, cubic_interp!, quadratic_interp!, constant_interp!,
            pchip_interp!, akima_interp!, cardinal_interp!,
        )
        f!(out, (x,), y, xq; hint = nothing)
        @test out == (ref .= f!(similar(ref), x, y, xq))          # in-place, bare vector
        f!(out, (x,), y, (xq,); hint = nothing)
        @test out == ref                                          # in-place, SoA
        hc = Ref(1); f!(out, (x,), y, xq; hint = (hc,))
        h1 = Ref(1); f!(similar(out), x, y, xq; hint = h1)
        @test hc[] == h1[] > 1                                     # in-place hint advances
    end

    # End-to-end: `interp!` with a `GridIdx` pins one axis and pre-slices to a 1-tuple
    # grid, threading `hint` into the collapse. This is the docs `unified_api.md` example.
    gx = collect(range(0.0, 5.0, length = 11))
    gy = collect(range(0.0, 5.0, length = 11))
    data = [xi + yj for xi in gx, yj in gy]
    output = zeros(5)
    q = collect(range(0.5, 4.5, length = 5))
    for m in (
            LinearInterp(), CubicInterp(), QuadraticInterp(), ConstantInterp(),
            PchipInterp(), AkimaInterp(), CardinalInterp(),
        )
        interp!(output, (gx, gy), data, (q, GridIdx(5)); method = m)
        @test all(isfinite, output)
    end
end

@testitem "1D interpolant accepts ND-style tuple queries" begin
    using FastInterpolations

    x = collect(1.0:10.0)
    y = @. sin(x)
    itp = linear_interp(x, y)

    xv = [2.3, 5.1, 8.8]
    out_t = similar(xv)
    out_s = similar(xv)

    # scalar 1-tuple query
    @test itp((3.7,)) == itp(3.7)
    # SoA in-place batch
    itp(out_t, (xv,)); itp(out_s, xv)
    @test out_t == out_s
    # SoA allocating batch
    @test itp((xv,)) == itp(xv)
    # per-axis kwarg unwrap on the query path
    @test itp((3.7,); deriv = (DerivOp(1),)) == itp(3.7; deriv = DerivOp(1))
end

# ── N=1 point containers reach the 1D engine (issue #204) ──
# The collapse's batch arm claims every batch container an ND caller may hand us —
# AoS `[(x,)]`, `Vector{Vector}`, shaped AoS, `GriddedQuery`, single-axis SoA — and
# forwards `_scalar_query(q)`: a numeric array passes through untouched, a point
# container becomes a lazy scalar view the 1D engine reads like a bare vector. So
# dimension-generic code written for N>=2 keeps working when N collapses to 1, AND it
# lands on the same lean 1D path a bare vector takes (bulk slope preparation for the
# local-Hermite family, not the per-query ND loop).
#
# Route witnesses that do not depend on `which`:
#   - values are `==` to the 1D call (same engine, same reads);
#   - explicit `coeffs = PreCompute()` on a local-Hermite method SUCCEEDS — the ND
#     engine rejects it (`_validate_nd_coeffs`), so success proves the 1D route.
@testitem "N=1 point containers reach the 1D engine" begin
    using FastInterpolations

    x = collect(range(0.0, 1.0, length = 8))
    y = @. sin(3x)
    qs = collect(range(0.05, 0.95, length = 7))

    aos = [(q,) for q in qs]            # Vector{Tuple{Float64}}  — the issue's shape
    aov = [[q] for q in qs]             # Vector{Vector{Float64}}
    gq = GriddedQuery((qs,))

    for (f, f!) in (
            (constant_interp, constant_interp!), (linear_interp, linear_interp!),
            (quadratic_interp, quadratic_interp!), (cubic_interp, cubic_interp!),
            (pchip_interp, pchip_interp!), (akima_interp, akima_interp!),
            (cardinal_interp, cardinal_interp!),
        )
        ref = f(x, y, qs)                              # 1D truth

        @test f((x,), y, aos) == ref                   # AoS allocating
        out = similar(ref)
        @test f!(out, (x,), y, aos) === out            # AoS in-place returns the sink
        @test out == ref
        @test f((x,), y, aov) == ref                   # Vector{Vector}
        @test f((x,), y, gq) == ref                    # GriddedQuery
        @test size(f((x,), y, gq)) == (length(qs),)
        # per-axis 1-tuple kwargs unwrap on the point-container route too
        @test f((x,), y, aos; deriv = (DerivOp(1),)) == f(x, y, qs; deriv = DerivOp(1))
    end

    # Shaped AoS keeps the query's shape (the ND shape-preservation contract). The
    # reference is the 1D BATCH on the same points — the scalar one-shot is a different
    # kernel and can differ by an ULP (`muladd` contraction, Julia-version dependent).
    M = reshape([(q,) for q in qs[1:6]], 2, 3)
    Mref = reshape(cubic_interp(x, y, qs[1:6]), 2, 3)
    @test size(cubic_interp((x,), y, M)) == (2, 3)
    @test cubic_interp((x,), y, M) == Mref
    outM = zeros(2, 3)
    cubic_interp!(outM, (x,), y, M)
    @test outM == Mref
    @test_throws DimensionMismatch cubic_interp!(zeros(6), (x,), y, M)   # exact-size sink

    # Route witness: the local-Hermite family accepts explicit PreCompute only on the
    # 1D path (ND rejects it with ArgumentError).
    for f in (pchip_interp, akima_interp, cardinal_interp)
        @test f((x,), y, aos; coeffs = PreCompute()) == f(x, y, qs; coeffs = PreCompute())
        @test f((x,), y, aos; coeffs = OnTheFly()) == f(x, y, qs; coeffs = OnTheFly())
    end
    # ...while explicit OnTheFly on cubic still selects the ND internals (no 1D equivalent).
    @test cubic_interp((x,), y, aos; coeffs = OnTheFly()) ≈ cubic_interp(x, y, qs) rtol = 1.0e-12

    # A ragged point container is rejected at the offending point, not truncated.
    @test_throws DimensionMismatch linear_interp((x,), y, [[0.2], [0.3, 0.4]])
    @test_throws DimensionMismatch linear_interp((x,), y, [(0.2, 0.3)])

    # GridIdx batches: the point-container route has exactly the bare route's outcome.
    # Today the 1D batch door has no GridIdx resolve, so under `--check-bounds=yes` both
    # throw (`<` on GridIdx) while with checks off the batch domain probe is elided and
    # both evaluate — the pin is "same outcome", not a value. Fixing that door lifts
    # both at once.
    outcome(q) = try
        (:value, linear_interp((x,), y, q))
    catch e
        (:error, typeof(e))
    end
    # Julia 1.10 cannot even build the bare `Vector{GridIdx}` literal (`vect` routes
    # through `convert(::Type{GridIdx{T}}, ::GridIdx)`, which has no 1.10-compatible
    # path) — a pre-existing quirk outside this contract, so the pin is skipped there.
    gbare = try
        [GridIdx(3), GridIdx(5)]
    catch
        nothing
    end
    gbare === nothing || @test outcome([(GridIdx(3),), (GridIdx(5),)]) == outcome(gbare)
end

@testitem "N=1 point containers through the unified interp/interp! API" begin
    using FastInterpolations

    x = collect(range(0.0, 1.0, length = 8))
    y = @. sin(3x)
    qs = collect(range(0.05, 0.95, length = 7))
    aos = [(q,) for q in qs]

    for (m, f) in (
            (ConstantInterp(), constant_interp), (LinearInterp(), linear_interp),
            (QuadraticInterp(), quadratic_interp), (CubicInterp(), cubic_interp),
            (PchipInterp(), pchip_interp), (AkimaInterp(), akima_interp),
            (CardinalInterp(), cardinal_interp),
        )
        ref = f(x, y, qs)
        @test interp((x,), y, aos; method = m) == ref
        @test interp((x,), y, qs; method = m) == ref
        @test interp((x,), y, (qs,); method = m) == ref
        out = similar(ref)
        @test interp!(out, (x,), y, aos; method = m) === out
        @test out == ref
        # the 1-tuple method form and per-axis kwargs are the ND spelling
        @test interp((x,), y, aos; method = (m,), deriv = (DerivOp(1),)) == f(x, y, qs; deriv = DerivOp(1))
    end

    # Route witness for the unified local-Hermite path: explicit PreCompute succeeds at
    # N=1 for bare, SoA and AoS alike (the ND engine rejects it), and the default
    # AutoCoeffs batch is the same bulk-slope result.
    for (m, f) in ((PchipInterp(), pchip_interp), (AkimaInterp(), akima_interp), (CardinalInterp(), cardinal_interp))
        ref = f(x, y, qs; coeffs = PreCompute())
        @test interp((x,), y, qs; method = m, coeffs = PreCompute()) == ref
        @test interp((x,), y, (qs,); method = m, coeffs = PreCompute()) == ref
        @test interp((x,), y, aos; method = m, coeffs = PreCompute()) == ref
        out = similar(ref)
        interp!(out, (x,), y, aos; method = m, coeffs = PreCompute())
        @test out == ref
        # bc still travels through the method object (inclusive periodic data)
        xp = collect(range(0.0, 2π, length = 9)); yp = cos.(xp)
        qsp = collect(range(0.3, 6.0, length = 7)); aosp = [(q,) for q in qsp]
        mp = FastInterpolations._replace_bc(m, PeriodicBC())
        @test interp((xp,), yp, aosp; method = mp) == f(xp, yp, qsp; bc = PeriodicBC())
    end
    # N=2 still rejects explicit PreCompute for local Hermite (unchanged ND contract).
    d2 = [sin(a) + cos(b) for a in x, b in x]
    @test_throws ArgumentError interp((x, x), d2, [(0.3, 0.4)]; method = PchipInterp(), coeffs = PreCompute())

    # Shaped AoS through the unified API keeps its shape.
    M = reshape(aos[1:6], 2, 3)
    @test size(interp((x,), y, M; method = PchipInterp())) == (2, 3)
    @test vec(interp((x,), y, M; method = PchipInterp())) == pchip_interp(x, y, qs[1:6])
end

# The point-container route costs nothing the bare route does not: no allocation past
# the output (the view is a stack struct over the caller's array) and a concretely
# inferred result. Measured through a `@noinline` barrier so the call — dispatch,
# adapter construction, kwarg unwrap — is inside the measurement.
@testitem "N=1 point-container route is allocation-free and inferred" setup = [AllocConstants] begin
    using FastInterpolations
    using Test: @inferred

    x = collect(range(0.0, 1.0, length = 40))
    y = @. sin(7x)
    qs = collect(range(0.02, 0.98, length = 25))
    aos = [(q,) for q in qs]
    out = similar(qs)

    # Every argument — including the grid tuple `(x,)` — is built OUTSIDE the measured
    # call: a tuple constructed inside the `@allocated` expression and handed to a
    # `@noinline` callee is boxed by the caller (16 B), on the bare route just the same.
    g = (x,)
    soa = (qs,)
    @noinline run!(f!::F, o, g, d, q) where {F} = (f!(o, g, d, q); nothing)
    @noinline urun!(o, g, d, q, m::M) where {M} = (interp!(o, g, d, q; method = m); nothing)

    for f! in (
            constant_interp!, linear_interp!, quadratic_interp!, cubic_interp!,
            pchip_interp!, cardinal_interp!, akima_interp!,
        )
        for q in (aos, soa, qs)                                             # warm every form
            run!(f!, out, g, y, q); run!(f!, out, g, y, q)
        end
        @test (@allocated run!(f!, out, g, y, aos)) <= ALLOC_THRESHOLD      # AoS in-place
        @test (@allocated run!(f!, out, g, y, soa)) <= ALLOC_THRESHOLD      # SoA in-place
        @test (@allocated run!(f!, out, g, y, qs)) <= ALLOC_THRESHOLD       # bare (reference)
        @test (@inferred f!(out, g, y, aos)) === out
    end
    for m in (PchipInterp(), CardinalInterp(), AkimaInterp(), CubicInterp(), LinearInterp())
        for q in (aos, qs)
            urun!(out, g, y, q, m); urun!(out, g, y, q, m)
        end
        @test (@allocated urun!(out, g, y, aos, m)) <= ALLOC_THRESHOLD      # unified, AoS
        @test (@allocated urun!(out, g, y, qs, m)) <= ALLOC_THRESHOLD       # unified, bare
        @test (@inferred interp!(out, g, y, aos; method = m)) === out
    end
    # allocating form: the result array is the only allocation
    @test (@inferred cubic_interp((x,), y, aos)) isa Vector{Float64}
end

# Regression pin: narrowing the collapse must NOT move the scalar-batch routes off the
# lean 1D path. These stay bit-identical; a `≈` here would hide exactly the regression
# this testitem exists to catch.
@testitem "N=1 scalar-batch routes stay bit-identical on the 1D path" begin
    using FastInterpolations

    x = collect(range(0.0, 1.0, length = 8))
    y = @. sin(3x)
    qs = collect(range(0.05, 0.95, length = 7))

    for (f, f!) in (
            (constant_interp, constant_interp!), (linear_interp, linear_interp!),
            (quadratic_interp, quadratic_interp!), (cubic_interp, cubic_interp!),
            (pchip_interp, pchip_interp!), (akima_interp, akima_interp!),
            (cardinal_interp, cardinal_interp!),
        )
        ref = f(x, y, qs)
        @test f((x,), y, qs) == ref          # bare Vector batch
        @test f((x,), y, (qs,)) == ref       # SoA
        @test f((x,), y, (0.5,)) == f(x, y, 0.5)   # scalar tuple
        out = similar(ref)
        f!(out, (x,), y, qs)
        @test out == ref                     # bare Vector, in-place
    end
end

# Issue #204 verbatim: dimension-generic ND code with ndim set to 1.
@testitem "issue #204: dimension-generic ND code at ndim=1" begin
    using FastInterpolations

    ndim, nquery, ndata = 1, 7, 5
    output = ones(nquery)
    grids = ntuple(n -> range(0, 1, length = ndata), ndim)
    data = [sum(sin, x) for x in Iterators.product(ntuple(n -> range(0, 1, length = ndata), ndim)...)]
    query = [ntuple(n -> y, ndim) for y in range(0, 1, length = nquery)]

    cubic_interp!(output, grids, data, query)
    @test all(isfinite, output)
    @test output == cubic_interp(only(grids), data, [q[1] for q in query])   # the 1D path itself
end
