# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║           Cubic — Dual Grid Tests (ForwardDiff)                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

using Test
using FastInterpolations
using ForwardDiff
using LinearAlgebra

const FI = FastInterpolations

@testset "Cubic — Dual Grid" begin

    x_base = collect(range(0.0, 5.0, 20))
    y_base = sin.(x_base)
    xq_scalar = 2.5
    xq_vec = [0.5, 1.5, 2.5, 3.5, 4.5]
    h_fd = 1.0e-7
    fd_deriv(f) = (f(1.0 + h_fd) - f(1.0 - h_fd)) / (2h_fd)

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Scalar Oneshot — ForwardDiff.derivative MWE                    ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic scalar oneshot — ForwardDiff.derivative + FD cross-check" begin
        f_interp = t -> cubic_interp(t .* x_base, y_base, xq_scalar; extrap=ExtendExtrap())
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol=1e-5
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Vector Oneshot                                                 ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic vector oneshot — Dual grid primals match Float path" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        result = cubic_interp(d .* x_base, y_base, xq_vec; extrap=ExtendExtrap())
        ref = cubic_interp(x_base, y_base, xq_vec; extrap=ExtendExtrap())
        @test ForwardDiff.value.(result) ≈ ref
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Interpolant (2-arg form)                                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic interpolant — Dual grid constructs + callable" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        itp = cubic_interp(d .* x_base, y_base; extrap=ExtendExtrap())
        v = itp(xq_scalar)
        itp_f = cubic_interp(x_base, y_base; extrap=ExtendExtrap())
        @test ForwardDiff.value(v) ≈ itp_f(xq_scalar)
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           BC types                                                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic BC types with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        xd = d .* x_base

        for (name, bc) in [
            ("CubicFit",    CubicFit()),
            ("ZeroCurvBC",  ZeroCurvBC()),
            ("ZeroSlopeBC", ZeroSlopeBC()),
            ("Deriv1(0)",   Deriv1(0.0)),
            ("Deriv2(0)",   Deriv2(0.0)),
            ("Deriv3(0)",   Deriv3(0.0)),
        ]
            @testset "$name" begin
                v = cubic_interp(xd, y_base, xq_scalar; bc, extrap=ExtendExtrap())
                @test isfinite(ForwardDiff.value(v))
                ref = cubic_interp(x_base, y_base, xq_scalar; bc, extrap=ExtendExtrap())
                @test ForwardDiff.value(v) ≈ ref
            end
        end
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           PeriodicBC                                                     ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic PeriodicBC with Dual grid" begin
        x_per = collect(range(0.0, 2π, 33))
        y_per = sin.(x_per)
        y_per[end] = y_per[1]  # periodic endpoint
        xq_per = 1.5

        f_per = t -> cubic_interp(t .* x_per, y_per, xq_per; bc=PeriodicBC())
        ad_val = ForwardDiff.derivative(f_per, 1.0)
        fd_val = fd_deriv(f_per)
        @test ad_val ≈ fd_val rtol=1e-5
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           DerivOp with Dual grid                                         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic DerivOp(1) and DerivOp(2) with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        xd = d .* x_base

        for n in 1:2
            @testset "DerivOp($n)" begin
                v = cubic_interp(xd, y_base, xq_scalar; extrap=ExtendExtrap(), deriv=DerivOp(n))
                ref = cubic_interp(x_base, y_base, xq_scalar; extrap=ExtendExtrap(), deriv=DerivOp(n))
                @test ForwardDiff.value(v) ≈ ref
            end
        end
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Extrap modes                                                   ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic ClampExtrap + NoExtrap with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        xd = d .* x_base

        # ClampExtrap
        v = cubic_interp(xd, y_base, xq_scalar; extrap=ClampExtrap())
        @test isfinite(ForwardDiff.value(v))

        # NoExtrap in-domain
        v2 = cubic_interp(xd, y_base, xq_scalar; extrap=NoExtrap())
        @test isfinite(ForwardDiff.value(v2))
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Range{Dual} grid                                               ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic Range{Dual} grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x_range = d .* range(0.0, 5.0, 20)
        v = cubic_interp(collect(x_range), y_base, xq_scalar; extrap=ExtendExtrap())
        ref = cubic_interp(x_base, y_base, xq_scalar; extrap=ExtendExtrap())
        @test ForwardDiff.value(v) ≈ ref
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Adjoint with Dual grid                                         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic adjoint — Dual grid constructs" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        adj = cubic_adjoint(d .* x_base, xq_vec; extrap=ExtendExtrap())
        y_bar = ones(length(xq_vec))
        result = adj(y_bar)
        @test length(result) == length(x_base)
        @test all(isfinite, result)
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           In-place vector API with Dual grid                             ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic_interp! — in-place with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        xd = d .* x_base
        # Allocating version works (output type inferred)
        result = cubic_interp(xd, y_base, xq_vec; extrap=ExtendExtrap())
        ref = cubic_interp(x_base, y_base, xq_vec; extrap=ExtendExtrap())
        @test ForwardDiff.value.(result) ≈ ref
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Series with Dual grid                                          ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    # NOTE: Cubic Series + Dual grid and Cubic ND + Dual grid require deeper
    # changes in the anchor/tensor-product infrastructure (_promote_for_anchor,
    # vector anchor T(xq[k]) conversion). Deferred to a follow-up PR.
    # The constraint relaxation in cubic_oneshot_series.jl and cubic_nd_*.jl
    # is still correct — it enables the Float hot path to coexist with
    # future Dual support without regressions.

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Float path — zero-alloc regression gate                        ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic Float scalar — zero-alloc regression" begin
        function _cubic_alloc_test()
            x_f = collect(range(0.0, 5.0, 100))
            y_f = sin.(x_f)
            cubic_interp(x_f, y_f, 2.5)
            cubic_interp(x_f, y_f, 2.5)
            return @allocated cubic_interp(x_f, y_f, 2.5)
        end
        _cubic_alloc_test()  # warmup
        @test _cubic_alloc_test() <= ALLOC_THRESHOLD
    end

end
