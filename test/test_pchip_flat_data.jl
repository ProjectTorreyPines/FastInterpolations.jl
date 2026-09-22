@testitem "PCHIP flat data: forward AD and adjoint" begin
    using FastInterpolations: _pchip_harmonic_mean
    using LinearAlgebra: dot
    using Random: MersenneTwister
    import ForwardDiff

    # Two adjacent control points holding the SAME value make a secant exactly zero — the
    # 0*0/0 the harmonic mean guards against — and both derivative paths used to miss that
    # guard: the forward's `iszero` inspected a Dual's PARTIALS (→ Dual(NaN, NaN), a NaN in
    # the VALUE); the adjoint's `sign(0) == sign(0)` took the active branch and divided by
    # δ² == 0. Ties are ordinary: clamped/quantized data, plateaus, shared optimizer bounds.

    x = collect(0.0:0.1:0.6)
    xq = collect(range(0.0, 0.6; length = 13))
    flat(p1, p2) = [0.0, 1.0, 2.0 + p1, 2.0 + p2, 2.0, 2.0, 5.0]   # plateau between two ramps

    @testset "kernel guard" begin
        D = ForwardDiff.Dual{:t}
        # value 0 with a live partial: the guard must still fire
        @test _pchip_harmonic_mean(3.0, 3.0, D(0.0, 8.0e-9), D(0.0, 0.0)) == D(0.0, 0.0)
        # one secant zero, the other live: den != 0, so the ordinary path returns the same zero
        @test _pchip_harmonic_mean(3.0, 3.0, D(0.0, 0.0), D(1.0, 2.0)) == D(0.0, 0.0)
        # non-degenerate input is untouched by the primal test
        @test _pchip_harmonic_mean(3.0, 3.0, 1.0, 2.0) ≈ 6 * 1.0 * 2.0 / (3.0 * 2.0 + 3.0 * 1.0)
    end

    @testset "forward under ForwardDiff" begin
        # INSIDE the plateau. Two live seeds are needed to trip the bug: with one, the secant
        # partials cancel on a uniform grid and restore `iszero(den)` — why 1-var AD passed.
        g(p) = pchip_interp(x, flat(p[1], p[2]); extrap = ExtendExtrap()).(xq)
        J = ForwardDiff.jacobian(p -> ForwardDiff.value.(g(p)), [0.0, 0.0])
        @test !any(isnan, J)
        # values are the plain-Float64 values (the AD path must not perturb the primal)
        @test ForwardDiff.value.(g([0.0, 0.0])) ≈
            pchip_interp(x, flat(0.0, 0.0); extrap = ExtendExtrap()).(xq)

        # Perturbing points OUTSIDE the plateau: smooth there, so AD must match finite
        # differences (the plateau still exists, exercising the guard).
        f(p) = pchip_interp(
            x, [0.0 + p[1], 1.0, 2.0, 2.0, 2.0, 2.0, 5.0 + p[2]];
            extrap = ExtendExtrap()
        ).(xq)
        J_ad = ForwardDiff.jacobian(f, [0.0, 0.0])
        h = 1.0e-6
        J_fd = hcat((f([h, 0.0]) .- f([-h, 0.0])) ./ 2h, (f([0.0, h]) .- f([0.0, -h])) ./ 2h)
        @test isapprox(J_ad, J_fd; rtol = 1.0e-5, atol = 1.0e-8)
    end

    @testset "adjoint on flat data" begin
        y = flat(0.0, 0.0)
        f_bar = pchip_adjoint(x, y, xq; extrap = ExtendExtrap())(ones(length(xq)))
        @test all(isfinite, f_bar)

        # The exactly-tied result must agree with the no-ties limit
        y_near = [0.0, 1.0, 2.0, 2.0 + 1.0e-9, 2.0 + 2.0e-9, 2.0 + 3.0e-9, 5.0]
        f_bar_near = pchip_adjoint(x, y_near, xq; extrap = ExtendExtrap())(ones(length(xq)))
        @test isapprox(f_bar, f_bar_near; rtol = 1.0e-6)

        # Dot-product identity <W*y, y_bar> == <y, W^T*y_bar>, on flat and fully-constant data
        rng = MersenneTwister(20240607)
        for yy in (flat(0.0, 0.0), fill(2.0, 7), [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
            y_bar = randn(rng, length(xq))
            lhs = dot(pchip_interp(x, yy; extrap = ExtendExtrap()).(xq), y_bar)
            rhs = dot(yy, pchip_adjoint(x, yy, xq; extrap = ExtendExtrap())(y_bar))
            @test isapprox(lhs, rhs; rtol = 1.0e-10)
        end

        # non-uniform grid, plateau not at the ends
        xn = [0.0, 0.5, 1.7, 2.0, 3.3, 4.0]
        yn = [1.0, 3.0, 3.0, 2.0, 2.0, 5.0]
        qn = collect(range(0.0, 4.0; length = 17))
        y_bar = randn(MersenneTwister(11), length(qn))
        @test isapprox(
            dot(pchip_interp(xn, yn; extrap = ExtendExtrap()).(qn), y_bar),
            dot(yn, pchip_adjoint(xn, yn, qn; extrap = ExtendExtrap())(y_bar)); rtol = 1.0e-10
        )
    end
end
