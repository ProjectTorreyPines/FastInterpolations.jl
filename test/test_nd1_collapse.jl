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
