# Secant fast-path helpers — value/type correctness across axis types.

@testitem "_get_inv_2cell accessor" setup = [AllocConstants] begin
    using FastInterpolations: _get_inv_2cell, _CachedVector, _to_float

    xv = [0.0, 0.5, 1.5, 4.0]            # non-uniform Vector
    cv = _CachedVector(xv)
    cr = _to_float(range(0.0, 10.0; length = 11), Float64)   # uniform range, h=1
    us = _to_float(1:11, Float64)                            # _UnitStep range

    # raw vector: inv of the 2-cell span x[3]-x[1]
    @test _get_inv_2cell(xv, 2) ≈ inv(xv[3] - xv[1])
    # cached vector: same value via cached widths
    @test _get_inv_2cell(cv, 2) ≈ inv(xv[3] - xv[1])
    # uniform range: span = 2h = 2.0
    @test _get_inv_2cell(cr, 5) ≈ inv(2.0)
    # unit-step: exactly 0.5, no division
    @test _get_inv_2cell(us, 5) === 0.5
end

@testitem "secant triad helpers" setup = [AllocConstants] begin
    using FastInterpolations: _forward_secant, _backward_secant, _centered_secant,
        _CachedVector, _to_float

    xv = [0.0, 0.5, 1.5, 4.0]
    yv = [1.0, 2.0, 0.0, 3.0]
    cv = _CachedVector(xv)
    us = _to_float(1:4, Float64)        # _UnitStep (h ≡ 1)

    # forward secant
    @test _forward_secant(xv, yv, 1) ≈ (yv[2] - yv[1]) / (xv[2] - xv[1])
    @test _forward_secant(cv, yv, 2) ≈ (yv[3] - yv[2]) / (xv[3] - xv[2])
    # backward at i ≡ forward at i-1 (exact identity, no recomputation)
    @test _backward_secant(xv, yv, 3) === _forward_secant(xv, yv, 2)
    # centered secant (2-cell span)
    @test _centered_secant(xv, yv, 2) ≈ (yv[3] - yv[1]) / (xv[3] - xv[1])

    # unit-step: division folds to identity ⇒ bit-identical to plain differences
    @test _forward_secant(us, yv, 1) === (yv[2] - yv[1])          # *one
    @test _centered_secant(us, yv, 2) === (yv[3] - yv[1]) * 0.5   # *0.5
end

@testitem "_pchip_harmonic_mean (3-div → 1-div rewrite)" setup = [AllocConstants] begin
    using FastInterpolations: _pchip_harmonic_mean
    using Random: MersenneTwister

    # Reference: original 3-division Fritsch–Carlson form.
    ref(w1, w2, δp, δc) = (w1 + w2) / (w1 / δp + w2 / δc)

    rng = MersenneTwister(11)
    for _ in 1:2000
        w1 = 5rand(rng) + 0.05
        w2 = 5rand(rng) + 0.05
        s = ifelse(rand(rng) < 0.5, -1.0, 1.0)        # same sign (monotone region)
        δp = s * (3rand(rng) + 1.0e-3)
        δc = s * (3rand(rng) + 1.0e-3)
        a = ref(w1, w2, δp, δc)
        b = _pchip_harmonic_mean(w1, w2, δp, δc)
        @test abs(a - b) <= 4 * eps(abs(a))            # ≤ a few ULP
    end

    # Equal secants ⇒ harmonic mean is that secant.
    @test _pchip_harmonic_mean(2.0, 3.0, 0.7, 0.7) ≈ 0.7 rtol = 1.0e-14

    # Degenerate flat data (δp == δc == 0): must be 0, NOT NaN (the 0·0/0 trap).
    @test _pchip_harmonic_mean(2.0, 3.0, 0.0, 0.0) == 0.0
    @test !isnan(_pchip_harmonic_mean(2.0, 3.0, 0.0, 0.0))
    # One zero secant ⇒ 0 (matches old Inf-limit behavior).
    @test _pchip_harmonic_mean(2.0, 3.0, 0.0, 1.5) == 0.0

    @test _pchip_harmonic_mean(2.0, 3.0, 0.5, 0.9) isa Float64
end

@testitem "secant fast-path: method equivalence + monotonicity" setup = [AllocConstants] begin
    using Random: MersenneTwister

    f(x) = sin(1.3x) + 0.2x
    grids = (
        ("unit-step", collect(1.0:1.0:20.0)),
        ("uniform", collect(range(0.0, 6.0, 24))),
        ("nonunif", sort(0.0 .+ 6.0 .* rand(MersenneTwister(7), 22))),
    )

    for (gname, x) in grids
        y = f.(x)
        q = gname == "unit-step" ? collect(range(1.0, 18.0, 50)) :
            collect(range(x[1] + 1.0e-3, x[end] - 1.0e-3, 50))
        for meth in (:pchip, :akima, :cardinal)
            interp = getfield(FastInterpolations, Symbol(meth, "_interp"))
            ot = interp(x, y, q; coeffs = FastInterpolations.OnTheFly())
            pc = interp(x, y, q; coeffs = FastInterpolations.PreCompute())
            @test ot ≈ pc atol = 1.0e-10
        end
    end

    # PCHIP monotonicity: monotone data ⇒ monotone interpolant (no overshoot).
    xm = collect(range(0.0, 10.0, 15))
    ym = cumsum(abs.(sin.(xm)) .+ 0.1)        # strictly increasing
    qm = collect(range(0.0, 10.0, 400))
    vm = pchip_interp(xm, ym, qm)
    @test all(diff(vm) .>= -1.0e-12)
end
