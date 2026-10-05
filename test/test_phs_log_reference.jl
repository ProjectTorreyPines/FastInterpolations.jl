# ============================================================================
# PHS log_reference — the log-transform reference keyword
# ============================================================================
# `log_reference` is `nothing` (off), a nonzero constant (used as given), or a
# callable ρ₀(q). Design: claudedocs/design/2026-09-30-phs-log-reference-api.md

@testitem "PHS log_reference — constant value does not change the result" begin
    x = range(0.0, π, 20)
    y = range(0.0, π, 20)
    rho = [1.5 + 0.4 * sin(xi) * cos(yj) for xi in x, yj in y]
    q = (1.1, 0.7)
    d1 = (DerivOp(1), EvalValue())
    d2 = (DerivOp(2), EvalValue())
    base = phs_interp((x, y), rho; stencil_size = 5, log_reference = 1.0)
    for c in (1.0e-3, 7.3, 1.0e3, 2)
        itp = phs_interp((x, y), rho; stencil_size = 5, log_reference = c)
        @test itp.transform.reference === c   # stored as given, no normalization
        @test itp(q) ≈ base(q) rtol = 1.0e-10
        @test isapprox(itp(q; deriv = d1), base(q; deriv = d1); rtol = 1.0e-8, atol = 1.0e-10)
        @test isapprox(itp(q; deriv = d2), base(q; deriv = d2); rtol = 1.0e-6, atol = 1.0e-8)
    end
end

@testitem "PHS log_reference — negative constant handles all-negative data" begin
    x = range(0.0, π, 20)
    y = range(0.0, π, 20)
    rho = [1.5 + 0.4 * sin(xi) * cos(yj) for xi in x, yj in y]
    q = (1.1, 0.7)
    d1 = (DerivOp(1), EvalValue())
    pos = phs_interp((x, y), rho; stencil_size = 5, log_reference = 1.0)
    neg = phs_interp((x, y), -rho; stencil_size = 5, log_reference = -2.0)
    @test neg(q) ≈ -pos(q) rtol = 1.0e-10
    @test neg(q; deriv = d1) ≈ -pos(q; deriv = d1) rtol = 1.0e-8
end

@testitem "PHS log_reference — invalid references rejected at build" begin
    x = range(0.0, π, 12)
    y = range(0.0, π, 12)
    rho = [1.5 + 0.4 * sin(xi) * cos(yj) for xi in x, yj in y]
    build(ref; data = rho) = phs_interp((x, y), data; stencil_size = 4, log_reference = ref)
    @test_throws ArgumentError build(0.0)                            # zero constant
    @test_throws ArgumentError build(true)                           # Bool: use `nothing` to disable
    @test_throws ArgumentError build(false)
    @test_throws ArgumentError build(-1.0)                           # sign differs from the data
    @test_throws ArgumentError build(1.0; data = rho .- 1.6)         # mixed-sign data, no exact zeros
    @test_throws ArgumentError build(1.0e-300; data = 1.0e10 .* rho) # ratio overflows to Inf
    @test_throws ArgumentError build(ones(12, 12))                   # arrays are not references
    @test_throws ArgumentError build("rho0")                         # not callable on a point
    @test_throws ArgumentError build((a, b) -> 1.0)                  # must take one NTuple point
    @test_throws ArgumentError build(p -> p[1] < 1.0 ? Inf : 1.0)    # non-finite ρ₀ at nodes
    @test_throws ArgumentError build(1.0; data = complex.(rho))      # needs real data
    @test_throws ArgumentError build(Inf)
    @test_throws ArgumentError build(NaN)
    @test_throws ArgumentError build(big"1.0e-400")                  # 0 at Float64 grid precision
    @test_throws ArgumentError build(p -> 1.0 + 0im)                 # ρ₀ must be real
    @test_throws ArgumentError build(p -> (1.0, 2.0))                # ρ₀ must be a scalar
    x32 = range(0.0f0, 1.0f0, 12)
    @test_throws ArgumentError phs_interp((x32,), ones(Float32, 12); log_reference = 1.0e-100)  # 0 as Float32
end

@testitem "PHS log_reference — interpolant reference is used natively" begin
    ρ0f(a, b) = exp(-hypot(a - 0.3, b - 0.2))
    exact(a, b) = ρ0f(a, b) * (1.5 + 0.3 * sin(2a) * cos(b))
    xf = range(0.0, 1.0, 81)                        # the reference comes from finer data
    ref = cubic_interp((xf, xf), [ρ0f(a, b) for a in xf, b in xf])
    x = range(0.0, 1.0, 21)
    itp = phs_interp((x, x), [exact(a, b) for a in x, b in x]; log_reference = ref)
    @test itp.transform.reference === ref           # deriv-aware: not wrapped
    q = (0.61, 0.47)
    @test itp(q) ≈ exact(q...) rtol = 1.0e-3
    h = 1.0e-5
    fd = (exact(q[1] + h, q[2]) - exact(q[1] - h, q[2])) / 2h
    @test itp(q; deriv = (DerivOp(1), EvalValue())) ≈ fd rtol = 2.0e-2
end

@testitem "PHS log_reference — value-only function answers value queries" begin
    x = range(0.0, 1.0, 12)
    rho = [exp(-a) * (1.5 + b) for a in x, b in x]
    itp = phs_interp((x, x), rho; log_reference = p -> exp(-p[1]))
    q = (0.43, 0.61)
    @test itp(q) ≈ exp(-q[1]) * (1.5 + q[2]) rtol = 1.0e-3
end

@testitem "PHS log_reference — a callable that takes deriv supplies its own derivatives" begin
    # A callable that accepts `deriv`, declared or through `kw...`, answers derivative
    # queries itself, as interpolants do. These references answer 0.25 for any
    # derivative, unlike the derivative of their value (0), so the result shows that
    # their own answer was used.
    struct DeclRef end
    (::DeclRef)(q; deriv = nothing) = deriv === nothing ? 1.0 : 0.25
    struct SlurpRef end
    (::SlurpRef)(q; kw...) = haskey(kw, :deriv) ? 0.25 : 1.0
    struct ValueRef end
    (::ValueRef)(q) = 1.0
    x = range(0.0, 1.0, 8)
    build(ref) = phs_interp((x, x), fill(2.0, 8, 8); stencil_size = 4, log_reference = ref)
    q = (0.43, 0.61)
    for ref in (DeclRef(), SlurpRef())
        itp = build(ref)
        @test itp.transform.reference === ref
        @test itp(q; deriv = (DerivOp(1), EvalValue())) ≈ 0.5 atol = 1.0e-10   # ρ₀′ G = 0.25 · 2
    end
    # A value-only callable is stored as given and answers value queries (§F6 covers
    # derivative queries).
    itpv = build(ValueRef())
    @test itpv.transform.reference === ValueRef()
    @test itpv(q) ≈ 2.0
    # N = 1: the 1-D interpolant takes tuple queries through `kw...`.
    ref1 = cubic_interp(x, exp.(-x))
    itp1 = phs_interp((x,), 2 .* exp.(-x); stencil_size = 4, log_reference = ref1)
    @test itp1.transform.reference === ref1
end

@testitem "PHS log_reference — Float32 data keeps Float32" begin
    x = range(0.0f0, 1.0f0, 12)
    rho = Float32[1.5 + 0.4 * sin(a) * cos(b) for a in x, b in x]
    itp = phs_interp((x, x), rho; stencil_size = 4, log_reference = 1.0)
    @test itp((0.3f0, 0.4f0)) isa Float32
    @test itp((0.3f0, 0.4f0); deriv = (DerivOp(1), EvalValue())) isa Float32
end

@testitem "PHS log_reference — one-shot forms forward it" begin
    x = range(0.0, π, 16)
    rho = [1.5 + 0.4 * sin(a) * cos(b) for a in x, b in x]
    kw = (; stencil_size = 5, blend_factor = 1.0, log_reference = 1.0)   # one-shot blend default differs (§R7)
    itp = phs_interp((x, x), rho; kw...)
    q = (1.1, 0.7)
    qs = ([1.1, 0.4], [0.7, 2.0])
    @test phs_interp((x, x), rho, q; kw...) ≈ itp(q) rtol = 1.0e-12
    @test phs_interp((x, x), rho, qs; kw...) ≈ itp(qs) rtol = 1.0e-12
    out = zeros(2)
    phs_interp!(out, (x, x), rho, qs; kw...)
    @test out ≈ itp(qs) rtol = 1.0e-12
end

@testitem "PHS log_reference — extreme but representable constant is accepted" begin
    x = range(0.0, π, 20)
    rho = [1.5 + 0.4 * sin(a) * cos(b) for a in x, b in x]
    q = (1.1, 0.7)
    base = phs_interp((x, x), rho; stencil_size = 5, log_reference = 1.0)
    tiny = phs_interp((x, x), rho; stencil_size = 5, log_reference = 1.0e-300)
    @test tiny(q) ≈ base(q) rtol = 1.0e-8   # measured max rel diff ~1e-10
end

@testitem "PHS log_reference — constant reference evaluation does not allocate" setup = [AllocConstants] begin
    x = range(0.0, π, 16)
    rho = [1.5 + 0.4 * sin(a) * cos(b) for a in x, b in x]
    itp = phs_interp((x, x), rho; stencil_size = 5, log_reference = 1.0)
    q = (1.1, 0.7)
    d1 = (DerivOp(1), EvalValue())
    d2 = (DerivOp(1), DerivOp(1))
    alloc_value(itp, q) = @allocated itp(q)
    alloc_deriv(itp, q, ops) = @allocated itp(q; deriv = ops)
    alloc_value(itp, q); alloc_deriv(itp, q, d1); alloc_deriv(itp, q, d2)   # warm code and caches
    @test alloc_value(itp, q) <= ND_ALLOC_THRESHOLD
    @test alloc_deriv(itp, q, d1) <= ND_ALLOC_THRESHOLD
    @test alloc_deriv(itp, q, d2) <= ND_ALLOC_THRESHOLD
end

@testitem "PHS log_reference — callable reference evaluation does not allocate" setup = [AllocConstants] begin
    # ρ₀ = exp(-Σ aₖ xₖ²) with hand-written derivatives; it reads every coordinate.
    struct GaussRef{N}
        a::NTuple{N, Float64}
    end
    function (g::GaussRef{N})(q; deriv = nothing) where {N}
        v = exp(-sum(ntuple(d -> g.a[d] * q[d]^2, Val(N))))
        deriv === nothing && return v
        o = map(deriv_order, deriv)
        sum(o) == 0 && return v
        d = something(findfirst(>(0), o))
        sum(o) == 1 && return -2 * g.a[d] * q[d] * v
        o[d] == 2 && return (4 * g.a[d]^2 * q[d]^2 - 2 * g.a[d]) * v
        e = something(findlast(>(0), o))
        return 4 * g.a[d] * g.a[e] * q[d] * q[e] * v
    end
    alloc_deriv(itp, q, ops) = @allocated itp(q; deriv = ops)
    EV = EvalValue()
    x = range(-1.0, 1.0, 21)
    g2 = GaussRef((1.0, 2.0))
    itp2 = phs_interp((x, x), [g2((a, b)) * (1.5 + 0.1a) for a in x, b in x]; log_reference = g2)
    q2 = (0.43, -0.21)
    for ops in ((DerivOp(1), EV), (DerivOp(2), EV), (DerivOp(1), DerivOp(1)))
        alloc_deriv(itp2, q2, ops)   # warm
        @test alloc_deriv(itp2, q2, ops) <= ND_ALLOC_THRESHOLD
    end
    # 3-D (the density use case)
    x3 = range(-1.0, 1.0, 9)
    g3 = GaussRef((1.0, 2.0, 3.0))
    itp3 = phs_interp((x3, x3, x3), [g3((a, b, c)) * (1.5 + 0.1a) for a in x3, b in x3, c in x3]; stencil_size = 4, log_reference = g3)
    q3 = (0.21, -0.33, 0.14)
    for ops in ((DerivOp(1), EV, EV), (EV, DerivOp(1), EV), (DerivOp(2), EV, EV), (DerivOp(1), EV, DerivOp(1)), (EV, DerivOp(1), DerivOp(1)))
        alloc_deriv(itp3, q3, ops)   # warm
        @test alloc_deriv(itp3, q3, ops) <= ND_ALLOC_THRESHOLD
    end
end

@testitem "PHS log_reference — transform evaluation docstring is attached" begin
    # A blank line between a docstring and its definition silently detaches it.
    meta = Base.Docs.meta(FastInterpolations)
    @test haskey(meta, Base.Docs.Binding(FastInterpolations, :_phs_eval_with_transform))
end
