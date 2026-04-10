# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║          LINEAR 1D — ForwardDiff.Dual grid support (Phase 1)               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# ForwardDiff-specific tests for Phase 1 Linear dual-grid support. Verifies that
# `Vector{ForwardDiff.Dual}` grids work end-to-end through the scalar callable
# and the scalar one-shot `linear_interp(x, y, xq)` API.
#
# Companion to `test/test_linear_dual_grid.jl` (which uses a custom duck type
# without ForwardDiff loaded).
#
# Run standalone:
#     cc-julia-test-runner . ext/test_linear_dual_grid.jl
#
# Phase 1 scope: scalar-only. Vector queries (`itp.(xq_vec)` that allocates)
# are Phase 2 territory and are NOT covered here.

using Test
using FastInterpolations
using ForwardDiff

const FI = FastInterpolations

@testset "Linear 1D — ForwardDiff.Dual grid (Phase 1)" begin

    @testset "Issue #81 MWE (scalar form)" begin
        # Reduced scalar variant of the MWE from
        # https://github.com/ProjectTorreyPines/FastInterpolations.jl/issues/81
        x = collect(1.0:0.1:10.0)
        y = log.(x)
        # Mid-interval query avoids the knot non-differentiability at grid points.
        xq = 2.55

        # The failing case on master: grid is Dual via `t .* x`
        ad_deriv = ForwardDiff.derivative(t -> linear_interp(t .* x, y, xq), 1.0)

        # Centered finite-difference reference
        h = 1e-6
        fd_deriv = (linear_interp((1.0 + h) .* x, y, xq) -
                    linear_interp((1.0 - h) .* x, y, xq)) / (2h)

        @test isapprox(ad_deriv, fd_deriv; atol = 1e-8)
    end

    @testset "Constructor with Vector{Dual}" begin
        x = collect(1.0:0.5:5.0)
        y = sin.(x)
        x_dual = ForwardDiff.Dual{:tag}.(x, ones(length(x)))

        itp = linear_interp(x_dual, y)

        # Struct type parameters carry the Dual through
        @test itp isa FI.LinearInterpolant
        @test eltype(itp.x) <: ForwardDiff.Dual
        @test eltype(itp.y) === Float64
        # Spacing cache is a real VectorSpacing{Dual} carrying partials
        @test itp.spacing isa FI.VectorSpacing{<:ForwardDiff.Dual}
        @test length(itp.spacing.h) == length(x) - 1
    end

    @testset "Scalar callable returns Dual with correct primal and partial" begin
        x = collect(1.0:0.1:10.0)
        y = log.(x)

        # Parameterization: x[i](ε) = x[i] + ε  (uniform shift).
        # Equivalently, ∂x[i]/∂ε = 1 for all i  →  partials = ones.
        x_dual = ForwardDiff.Dual{:tag}.(x, ones(length(x)))
        itp = linear_interp(x_dual, y)

        xq = 2.55  # mid-interval (avoids knot non-differentiability)
        r = itp(xq)
        @test r isa ForwardDiff.Dual

        # Primal matches ordinary linear interpolation value
        itp_float = linear_interp(x, y)
        @test ForwardDiff.value(r) ≈ itp_float(xq)

        # Partial reference: centered FD over the SAME uniform-shift parameter,
        # i.e. `linear_interp(x ± h, y, xq)` — NOT `linear_interp((1±h)*x, ...)`,
        # which would correspond to a different parameterization ∂(t*x)/∂t = x[i].
        h = 1e-6
        fd = (linear_interp(x .+ h, y, xq) -
              linear_interp(x .- h, y, xq)) / (2h)
        @test isapprox(ForwardDiff.partials(r)[1], fd; atol = 1e-8)
    end

    @testset "One-shot scalar matches callable form" begin
        x = collect(1.0:0.1:10.0)
        y = log.(x)
        x_dual = ForwardDiff.Dual{:tag}.(x, ones(length(x)))
        itp = linear_interp(x_dual, y)

        for xq in (1.25, 2.55, 5.05, 7.75, 9.45)
            a = itp(xq)
            b = linear_interp(x_dual, y, xq)
            @test ForwardDiff.value(a) == ForwardDiff.value(b)
            @test ForwardDiff.partials(a) == ForwardDiff.partials(b)
        end
    end

    @testset "Endpoint query (at knot)" begin
        x = collect(1.0:0.1:5.0)
        y = x .^ 2
        x_dual = ForwardDiff.Dual{:tag}.(x, ones(length(x)))
        # ExtendExtrap here is intentional. ForwardDiff defines `<` on Dual such
        # that `1.0 < Dual(1.0, +1.0) == true` (lexicographic tiebreak on the
        # partial). So a Float knot query 1.0 against a Dual grid whose first
        # point has positive partial would fail a strict `NoExtrap` domain check,
        # even though the primal is on the boundary. ExtendExtrap sidesteps that
        # and lets us verify the knot-value behavior directly.
        itp = linear_interp(x_dual, y; extrap = ExtendExtrap())

        # Query exactly at the first knot (primal): value = y[1].
        r_lo = itp(first(x))
        @test ForwardDiff.value(r_lo) ≈ first(y)
        # At a grid point the linear interpolant is C⁰ but not C¹; either one-sided
        # derivative is a legitimate answer. We only assert the primal is correct.

        # Query exactly at the last knot (primal): value = y[end].
        r_hi = itp(last(x))
        @test ForwardDiff.value(r_hi) ≈ last(y)
    end

    @testset "Non-uniform Dual grid" begin
        x = [1.0, 1.3, 2.7, 4.0, 6.8, 10.0]
        y = cos.(x)
        # Non-trivial per-point partials — each grid point has a different sensitivity.
        partials = [0.1, 0.3, -0.2, 0.5, 0.0, 0.8]
        x_dual = [ForwardDiff.Dual{:tag}(x[i], partials[i]) for i in eachindex(x)]
        itp = linear_interp(x_dual, y)

        xq = 3.25  # lies in interval [2.7, 4.0]
        r = itp(xq)
        @test r isa ForwardDiff.Dual

        # Cross-check the partial via ForwardDiff.derivative on a closure that
        # rebuilds the grid from a parameter ε: x_i(ε) = x_i + ε*partials[i].
        ad_deriv = ForwardDiff.derivative(1.0) do ε
            x_shifted = [x[i] + (ε - 1.0) * partials[i] for i in eachindex(x)]
            linear_interp(x_shifted, y, xq)
        end

        @test isapprox(ForwardDiff.partials(r)[1], ad_deriv; atol = 1e-10)
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Edge Cases: on-grid, domain boundary, beyond domain          ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    #
    # ForwardDiff's lexicographic ordering (`Dual(x, +p) > x` when p > 0) means
    # that the sign of the grid partial at a knot determines whether a Float query
    # at that knot is seen as "inside" or "outside" the domain:
    #
    #   partial > 0  →  Dual(x, +p) > x  →  Float xq == x seen as < grid endpoint  →  OOB_LEFT possible
    #   partial < 0  →  Dual(x, -p) < x  →  Float xq == x seen as > grid endpoint  →  OOB_RIGHT possible
    #   partial == 0 →  Dual(x, 0) == x  →  Float comparison, same as pure Float path
    #
    # These tests exercise all three partial-sign regimes at both domain endpoints,
    # with on-grid interior knots, and with queries truly beyond the domain.

    @testset "On-grid query: partial sign controls interval selection" begin
        # Simple 3-point grid: values chosen so left and right slopes differ.
        x = [1.0, 2.0, 3.0]
        y = [10.0, 20.0, 15.0]  # slope_left = 10, slope_right = -5

        for (label, p, expected_slope) in [
            # positive partial → Dual(2, +p) > 2 → LEFT interval selected
            ("positive partial", +1.0, 10.0),
            # negative partial → Dual(2, -p) < 2 → RIGHT interval selected
            ("negative partial", -1.0, -5.0),
            # zero partial → same as Float, RIGHT interval (x[mid] <= xq is true)
            ("zero partial",      0.0, -5.0),
        ]
            @testset "$label" begin
                x_dual = [ForwardDiff.Dual{:tag}(xi, p) for xi in x]
                itp = linear_interp(x_dual, y; extrap = ExtendExtrap())

                r = itp(2.0)  # query at exact knot x[2]
                @test ForwardDiff.value(r) ≈ 20.0   # primal is always y[2]

                # deriv=1 reveals which interval's slope was used
                d = itp(2.0; deriv = DerivOp(1))
                @test ForwardDiff.value(d) ≈ expected_slope atol = 1e-12
            end
        end
    end

    @testset "Domain boundary: partial sign determines NoExtrap behavior" begin
        x = [1.0, 2.0, 3.0]
        y = [10.0, 20.0, 15.0]

        @testset "Left boundary (first grid point)" begin
            # positive partial: Dual(1, +1) > 1.0 → query 1.0 is OOB_LEFT → throw
            x_pos = [ForwardDiff.Dual{:tag}(xi, +1.0) for xi in x]
            itp_pos = linear_interp(x_pos, y)  # NoExtrap default
            @test_throws DomainError itp_pos(1.0)

            # negative partial: Dual(1, -1) < 1.0 → query 1.0 is IN_DOMAIN → works
            x_neg = [ForwardDiff.Dual{:tag}(xi, -1.0) for xi in x]
            itp_neg = linear_interp(x_neg, y)
            r = itp_neg(1.0)
            @test ForwardDiff.value(r) ≈ 10.0

            # zero partial: Dual(1, 0) == 1.0 → same as Float, IN_DOMAIN → works
            x_zero = [ForwardDiff.Dual{:tag}(xi, 0.0) for xi in x]
            itp_zero = linear_interp(x_zero, y)
            r = itp_zero(1.0)
            @test ForwardDiff.value(r) ≈ 10.0
        end

        @testset "Right boundary (last grid point)" begin
            # positive partial: Dual(3, +1) > 3.0, so 3.0 > Dual(3, +1) = false
            # → NOT OOB_RIGHT → domain check passes → works.
            # (The grid "expanded" rightward in the Dual sense, so query is inside.)
            x_pos = [ForwardDiff.Dual{:tag}(xi, +1.0) for xi in x]
            itp_pos = linear_interp(x_pos, y)
            r = itp_pos(3.0)
            @test ForwardDiff.value(r) ≈ 15.0

            # negative partial: Dual(3, -1) < 3.0, so 3.0 > Dual(3, -1) = true
            # → OOB_RIGHT → throw. (The grid "contracted" rightward, query fell outside.)
            x_neg = [ForwardDiff.Dual{:tag}(xi, -1.0) for xi in x]
            itp_neg = linear_interp(x_neg, y)
            @test_throws DomainError itp_neg(3.0)

            # zero partial: same as Float, IN_DOMAIN → works
            x_zero = [ForwardDiff.Dual{:tag}(xi, 0.0) for xi in x]
            itp_zero = linear_interp(x_zero, y)
            r = itp_zero(3.0)
            @test ForwardDiff.value(r) ≈ 15.0
        end
    end

    @testset "Beyond domain: extrapolation modes with Dual grid" begin
        x = [1.0, 2.0, 3.0]
        y = [10.0, 20.0, 15.0]
        x_dual = [ForwardDiff.Dual{:tag}(xi, 1.0) for xi in x]

        @testset "NoExtrap throws for truly-out-of-domain queries" begin
            itp = linear_interp(x_dual, y)
            @test_throws DomainError itp(0.5)   # well below domain
            @test_throws DomainError itp(3.5)   # well above domain
        end

        @testset "ExtendExtrap: linear extrapolation from boundary interval" begin
            itp = linear_interp(x_dual, y; extrap = ExtendExtrap())

            # Below domain: extrapolates from first interval [x[1], x[2]]
            r_lo = itp(0.5)
            @test r_lo isa ForwardDiff.Dual
            # Primal: y[1] + (0.5 - 1.0) * slope_first = 10 + (-0.5)*10 = 5
            @test ForwardDiff.value(r_lo) ≈ 5.0

            # Above domain: extrapolates from last interval [x[2], x[3]]
            r_hi = itp(3.5)
            @test r_hi isa ForwardDiff.Dual
            # Primal: y[2] + (3.5 - 2.0) * slope_last = 20 + 1.5*(-5) = 12.5
            @test ForwardDiff.value(r_hi) ≈ 12.5
        end

        @testset "ClampExtrap: boundary value for out-of-domain" begin
            itp = linear_interp(x_dual, y; extrap = ClampExtrap())
            r_lo = itp(0.5)
            r_hi = itp(3.5)
            # ClampExtrap returns the boundary y value (not a Dual with grid partials —
            # the clamped extrapolation path returns _promote_extrap_val which is
            # y_bnd + 0*xq = Float since y is Float and xq is Float).
            @test r_lo ≈ first(y)
            @test r_hi ≈ last(y)
        end

        @testset "FillExtrap: fill value for out-of-domain" begin
            itp = linear_interp(x_dual, y; extrap = FillExtrap(-999.0))
            @test itp(0.5) ≈ -999.0
            @test itp(3.5) ≈ -999.0
        end
    end

    @testset "On-grid query with per-point partial control" begin
        # Demonstrate that partial values are fully user-controlled per grid point.
        # Only x[2] is "sensitive" (partial = 3.7); endpoints have partial = 0.
        # This ensures the derivative only carries sensitivity through the grid
        # points that participate in the active interval.
        x = [1.0, 2.0, 3.0, 4.0]
        y = [0.0, 1.0, 4.0, 9.0]
        partials = [0.0, 3.7, 0.0, 0.0]
        x_dual = [ForwardDiff.Dual{:tag}(x[i], partials[i]) for i in eachindex(x)]
        itp = linear_interp(x_dual, y; extrap = ExtendExtrap())

        @testset "Mid-interval query in [x[2], x[3]] — both endpoints matter" begin
            # interval [Dual(2, 3.7), Dual(3, 0)]
            # h(ε) = (3 + 0*ε) - (2 + 3.7*ε) = 1 - 3.7ε
            # α(ε) = (2.5 - 2 - 3.7ε) / (1 - 3.7ε) = (0.5 - 3.7ε)/(1 - 3.7ε)
            # dα/dε|₀ = (-3.7 * 1 - 0.5 * (-3.7)) / 1² = -3.7 + 1.85 = -1.85
            # value = y[2] + α*(y[3]-y[2]) = 1 + α*3
            # dvalue/dε = dα/dε * 3 = -1.85 * 3 = -5.55
            r = itp(2.5)
            @test ForwardDiff.value(r) ≈ 2.5  # 1 + 0.5*3
            @test isapprox(ForwardDiff.partials(r)[1], -5.55; atol = 1e-12)
        end

        @testset "Mid-interval in [x[3], x[4]] — both partials zero → zero derivative" begin
            # Both x[3] and x[4] have partial = 0 → no grid sensitivity here.
            r = itp(3.5)
            @test ForwardDiff.value(r) ≈ 6.5  # 4 + 0.5*5
            @test isapprox(ForwardDiff.partials(r)[1], 0.0; atol = 1e-12)
        end

        @testset "On-grid at x[2] — active partial, interval selection by sign" begin
            # x[2] has partial = +3.7 (positive) → LEFT interval [x[1], x[2]] selected.
            # In LEFT interval, xL = Dual(1, 0), xR = Dual(2, 3.7)
            # dL = 2.0 - Dual(1, 0) = Dual(1, 0); inv_h = inv(Dual(2,3.7) - Dual(1,0))
            #                                             = inv(Dual(1, 3.7)) = Dual(1, -3.7)
            # α = dL * inv_h = Dual(1,0) * Dual(1,-3.7) = Dual(1, -3.7)
            # value = y[1] + α*(y[2]-y[1]) = 0 + Dual(1,-3.7)*1 = Dual(1, -3.7)
            # So partial = -3.7.
            r = itp(2.0)
            @test ForwardDiff.value(r) ≈ 1.0  # y[2]
            @test isapprox(ForwardDiff.partials(r)[1], -3.7; atol = 1e-12)
        end

        @testset "On-grid at x[3] — zero partial, behaves like Float" begin
            # x[3] has partial = 0 → Dual(3, 0) == 3.0 in lexicographic.
            # Same interval selection as Float path (RIGHT: [x[3], x[4]]).
            # xL = Dual(3, 0), xR = Dual(4, 0), dL = 0 → α = 0 → partial = 0.
            r = itp(3.0)
            @test ForwardDiff.value(r) ≈ 4.0  # y[3]
            @test isapprox(ForwardDiff.partials(r)[1], 0.0; atol = 1e-12)
        end
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║              Range{Dual} grid — _CachedRange + DirectSearch O(1)       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    #
    # StepRangeLen{Dual} is created via `t .* range` (broadcast), NOT `t * range`
    # (scalar mult), because the latter hits a Julia Base MethodError in
    # `twiceprecision(TwicePrecision{Dual}, Int)` — a Base limitation, not ours.

    @testset "Range{Dual} construction → _CachedRange{Dual}" begin
        x_range = 1.0:0.1:5.0
        y = sin.(collect(x_range))
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x_dual_range = d .* x_range  # StepRangeLen{Dual, TwicePrecision{Dual}, ...}

        itp = linear_interp(x_dual_range, y)

        # Stored as _CachedRange{Dual}, not the raw StepRangeLen
        @test itp.x isa FI._CachedRange{<:ForwardDiff.Dual}
        # ScalarSpacing{Dual} — uniform grid, O(1) memory
        @test itp.spacing isa FI.ScalarSpacing{<:ForwardDiff.Dual}
    end

    @testset "Range{Dual} scalar eval matches Vector{Dual} path" begin
        x_range = 1.0:0.1:5.0
        y = sin.(collect(x_range))
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x_dual_range = d .* x_range
        x_dual_vec = collect(x_dual_range)

        itp_range = linear_interp(x_dual_range, y)
        itp_vec = linear_interp(x_dual_vec, y)

        for xq in (1.25, 2.55, 3.75, 4.55)
            vr = itp_range(xq)
            vv = itp_vec(xq)
            @test ForwardDiff.value(vr) ≈ ForwardDiff.value(vv)
            # Partials may differ by a few ULPs due to TwicePrecision rounding
            @test isapprox(ForwardDiff.partials(vr)[1], ForwardDiff.partials(vv)[1]; atol = 1e-10)
        end
    end

    @testset "Range{Dual} MWE: ForwardDiff.derivative through range grid" begin
        x_range = 1.0:0.1:10.0
        y = log.(collect(x_range))
        xq = 2.55

        # t .* range (broadcast) creates StepRangeLen{Dual} — this works.
        # NOTE: t * range (scalar mult) fails with Julia Base MethodError in
        # twiceprecision(TwicePrecision{Dual}, Int) — this is a Base limitation.
        ad = ForwardDiff.derivative(t -> linear_interp(t .* x_range, y, xq; extrap = ExtendExtrap()), 1.0)

        # Cross-check with Vector path
        ad_vec = ForwardDiff.derivative(t -> linear_interp(t .* collect(x_range), y, xq), 1.0)
        @test isapprox(ad, ad_vec; atol = 1e-10)

        # Cross-check with FD
        h = 1e-6
        fd = (linear_interp(collect((1+h) .* x_range), y, xq) -
              linear_interp(collect((1-h) .* x_range), y, xq)) / (2h)
        @test isapprox(ad, fd; atol = 1e-8)
    end

    @testset "Range{Dual} — all range source types" begin
        # Every AbstractRange variant that can produce a Dual-eltype range.
        # All should normalize to _CachedRange{Dual} + ScalarSpacing{Dual}.
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)

        range_cases = [
            # (label,           Dual range expression,        y values)
            ("StepRangeLen",    d .* (1.0:0.5:5.0),          sin.(collect(1.0:0.5:5.0))),
            ("LinRange",        d .* range(1.0, 5.0, 9),     sin.(range(1.0, 5.0, 9))),
            ("UnitRange .*",    d .* (1:5),                   sin.(Float64.(1:5))),
            ("StepRange .*",    d .* (1:2:9),                 sin.(Float64.(1:2:9))),
            ("UnitRange *",     d * (1:5),                    sin.(Float64.(1:5))),
            ("StepRange *",     d * (1:2:9),                  sin.(Float64.(1:2:9))),
        ]

        for (label, x_dual, y) in range_cases
            @testset "$label" begin
                itp = linear_interp(x_dual, y; extrap = ExtendExtrap())

                # All normalized to _CachedRange{Dual}
                @test itp.x isa FI._CachedRange{<:ForwardDiff.Dual}
                @test itp.spacing isa FI.ScalarSpacing{<:ForwardDiff.Dual}

                # Scalar eval returns Dual.
                # Use a mid-interval query to avoid exact-knot tiebreak differences
                # between DirectSearch (primal trunc → RIGHT) and BinarySearch
                # (lexicographic → LEFT). At mid-interval, both give identical results.
                lo_p = ForwardDiff.value(first(itp.x))
                hi_p = ForwardDiff.value(last(itp.x))
                xq = lo_p + (hi_p - lo_p) * 0.37  # irrational fraction avoids knots
                r = itp(xq)
                @test r isa ForwardDiff.Dual

                # Compare with collected Vector{Dual} path
                itp_vec = linear_interp(collect(x_dual), y; extrap = ExtendExtrap())
                r_vec = itp_vec(xq)
                @test ForwardDiff.value(r) ≈ ForwardDiff.value(r_vec)
                @test isapprox(ForwardDiff.partials(r)[1], ForwardDiff.partials(r_vec)[1]; atol = 1e-10)
            end
        end
    end

    @testset "Range{Dual} — exact-knot behavior" begin
        # At an exact knot, DirectSearch (Range path, primal-based trunc) always
        # picks the RIGHT interval, while BinarySearch (Vector path, lexicographic)
        # picks LEFT when the grid partial is positive.
        # Both are legitimate sub-gradient choices; we verify each is consistent
        # with its own one-sided FD reference.

        x_range = 1.0:0.5:5.0
        y = [0.0, 1.0, 4.0, 2.0, 5.0, 3.0, 7.0, 1.5, 6.0]
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x_dual_range = d .* x_range
        x_dual_vec = collect(x_dual_range)
        xq_knot = 3.0  # = x[5], an interior knot

        itp_range = linear_interp(x_dual_range, y; extrap = ExtendExtrap())
        itp_vec = linear_interp(x_dual_vec, y; extrap = ExtendExtrap())

        # Primal values should be identical (y[5] at the knot)
        r_range = itp_range(xq_knot)
        r_vec = itp_vec(xq_knot)
        @test ForwardDiff.value(r_range) ≈ y[5]
        @test ForwardDiff.value(r_vec) ≈ y[5]

        # Slopes on either side of the knot
        slope_left = (y[5] - y[4]) / 0.5    # left interval [x[4], x[5]]
        slope_right = (y[6] - y[5]) / 0.5   # right interval [x[5], x[6]]

        # deriv=1 reveals which slope each path selected
        d_range = itp_range(xq_knot; deriv = DerivOp(1))
        d_vec = itp_vec(xq_knot; deriv = DerivOp(1))

        # Range/DirectSearch → RIGHT interval → right slope
        @test ForwardDiff.value(d_range) ≈ slope_right
        # Vector/BinarySearch with positive partial → LEFT interval → left slope
        @test ForwardDiff.value(d_vec) ≈ slope_left

        # AD partials at exact knots: verify against BOTH analytic chain rule
        # (exact) and one-sided FD (numerical cross-check).
        #
        # Analytic (quotient rule on α(t) = (xq - t*x[k]) / (t*step)):
        #   dα/dt|_{t=1} = -xq / step = -3.0 / 0.5 = -6
        #
        # Range/DirectSearch → RIGHT interval [x[5], x[6]]:
        #   ∂f/∂t = dα/dt * (y[6]-y[5]) = -6 * (3-5) = 12
        @test ForwardDiff.partials(r_range)[1] ≈ 12.0
        # Vector/BinarySearch → LEFT interval [x[4], x[5]]:
        #   ∂f/∂t = dα/dt * (y[5]-y[4]) = -6 * (5-2) = -18
        @test ForwardDiff.partials(r_vec)[1] ≈ -18.0

        # One-sided FD cross-check: each AD should match one of the two one-sided FDs.
        h = 1e-7
        f_base_r = linear_interp(x_range, y, xq_knot; extrap = ExtendExtrap())
        fd_fwd_r = (linear_interp((1+h) .* x_range, y, xq_knot; extrap = ExtendExtrap()) - f_base_r) / h
        fd_bwd_r = (f_base_r - linear_interp((1-h) .* x_range, y, xq_knot; extrap = ExtendExtrap())) / h
        @test (isapprox(ForwardDiff.partials(r_range)[1], fd_fwd_r; atol = 1e-3) ||
               isapprox(ForwardDiff.partials(r_range)[1], fd_bwd_r; atol = 1e-3))

        f_base_v = linear_interp(collect(x_range), y, xq_knot; extrap = ExtendExtrap())
        fd_fwd_v = (linear_interp(collect((1+h) .* x_range), y, xq_knot; extrap = ExtendExtrap()) - f_base_v) / h
        fd_bwd_v = (f_base_v - linear_interp(collect((1-h) .* x_range), y, xq_knot; extrap = ExtendExtrap())) / h
        @test (isapprox(ForwardDiff.partials(r_vec)[1], fd_fwd_v; atol = 1e-3) ||
               isapprox(ForwardDiff.partials(r_vec)[1], fd_bwd_v; atol = 1e-3))
    end

    @testset "Range{Dual} one-shot scalar" begin
        x_range = 1.0:0.5:5.0
        y = cos.(collect(x_range))
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x_dual_range = d .* x_range

        itp = linear_interp(x_dual_range, y; extrap = ExtendExtrap())
        for xq in (1.25, 2.75, 4.5)
            v_call = itp(xq)
            v_shot = linear_interp(x_dual_range, y, xq; extrap = ExtendExtrap())
            @test ForwardDiff.value(v_call) == ForwardDiff.value(v_shot)
            @test ForwardDiff.partials(v_call) == ForwardDiff.partials(v_shot)
        end
    end

    @testset "Type stability via @inferred" begin
        x = collect(1.0:0.1:10.0)
        y = sin.(x)
        x_dual = ForwardDiff.Dual{:tag}.(x, ones(length(x)))
        itp = linear_interp(x_dual, y)

        # Scalar callable must return a concrete Dual type, not an abstract Union
        @test_nowarn @inferred itp(2.55)
        @test_nowarn @inferred linear_interp(x_dual, y, 2.55)
    end

    @testset "Regression: Float grid path through the same API" begin
        # After removing the Real wrapper, Float scalar path still flows cleanly
        # through the unified core method and preserves zero-alloc behavior.
        x = collect(1.0:0.1:10.0)
        y = sin.(x)

        itp = linear_interp(x, y)
        itp(5.55)  # warmup
        @test (@allocations itp(5.55)) == 0
    end
end
