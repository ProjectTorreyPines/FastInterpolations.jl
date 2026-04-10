# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║           Hermite Family — Dual Grid Tests (ForwardDiff)                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Tests that hermite_interp, pchip_interp, cardinal_interp, akima_interp
# accept Vector{Dual} and Range{Dual} grids — enabling
# ForwardDiff.derivative(t -> method_interp(t .* x, y, xq), 1.0).
#
# Follows the same pattern as test_linear_dual_grid.jl.

using Test
using FastInterpolations
using ForwardDiff
using LinearAlgebra

const FI = FastInterpolations

@testset "Hermite Family — Dual Grid" begin

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║                    Shared test data                                    ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    x_base = collect(range(0.0, 2π, 30))
    y_base = sin.(x_base)
    dy_base = cos.(x_base)  # exact slopes for hermite_interp
    xq_scalar = 2.5
    xq_vec = [1.0, 2.0, 3.0, 4.0, 5.0]
    h_fd = 1.0e-7  # FD step

    # Helper: finite-difference derivative of f(t) at t=1
    fd_deriv(f) = (f(1.0 + h_fd) - f(1.0 - h_fd)) / (2h_fd)

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           hermite_interp (user-supplied slopes)                        ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "hermite_interp — scalar oneshot with Dual grid" begin
        f = t -> hermite_interp(t .* x_base, y_base, dy_base, xq_scalar; extrap = ExtendExtrap())
        v = f(ForwardDiff.Dual{:tag}(1.0, 1.0))
        @test v isa ForwardDiff.Dual
        @test ForwardDiff.value(v) ≈ hermite_interp(x_base, y_base, dy_base, xq_scalar; extrap = ExtendExtrap())
        @test isapprox(ForwardDiff.partials(v)[1], fd_deriv(f); atol = 1.0e-5)
    end

    @testset "hermite_interp — ForwardDiff.derivative MWE" begin
        deriv = ForwardDiff.derivative(
            t -> hermite_interp(t .* x_base, y_base, dy_base, xq_scalar; extrap = ExtendExtrap()), 1.0
        )
        fd = fd_deriv(t -> hermite_interp(t .* x_base, y_base, dy_base, xq_scalar; extrap = ExtendExtrap()))
        @test isapprox(deriv, fd; atol = 1.0e-5)
    end

    @testset "hermite_interp — interpolant with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        itp = hermite_interp(d .* x_base, y_base, dy_base; extrap = ExtendExtrap())
        @test itp isa FI.CubicHermiteInterpolant1D
        v = itp(xq_scalar)
        @test v isa ForwardDiff.Dual
        itp_f = hermite_interp(x_base, y_base, dy_base; extrap = ExtendExtrap())
        @test ForwardDiff.value(v) ≈ itp_f(xq_scalar)
    end

    @testset "hermite_interp — vector oneshot with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        result = hermite_interp(d .* x_base, y_base, dy_base, xq_vec; extrap = ExtendExtrap())
        @test eltype(result) <: ForwardDiff.Dual
        ref = hermite_interp(x_base, y_base, dy_base, xq_vec; extrap = ExtendExtrap())
        @test ForwardDiff.value.(result) ≈ ref
    end

    @testset "hermite_interp — adjoint with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x_dual = d .* x_base
        adj = hermite_adjoint(x_dual, xq_vec; extrap = ExtendExtrap())
        @test adj isa FI.HermiteAdjoint1D
        y_bar = randn(length(xq_vec))
        f_bar = adj(y_bar)
        @test eltype(f_bar) <: ForwardDiff.Dual
    end

    @testset "hermite_interp — integrate with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        itp = hermite_interp(d .* x_base, y_base, dy_base; extrap = ExtendExtrap())
        I = integrate(itp, x_base[5], x_base[20])
        @test I isa ForwardDiff.Dual
        itp_f = hermite_interp(x_base, y_base, dy_base; extrap = ExtendExtrap())
        @test ForwardDiff.value(I) ≈ integrate(itp_f, x_base[5], x_base[20])
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           pchip_interp                                                 ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "pchip_interp — scalar oneshot with Dual grid" begin
        f = t -> pchip_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap())
        v = f(ForwardDiff.Dual{:tag}(1.0, 1.0))
        @test v isa ForwardDiff.Dual
        @test ForwardDiff.value(v) ≈ pchip_interp(x_base, y_base, xq_scalar; extrap = ExtendExtrap())
        @test isapprox(ForwardDiff.partials(v)[1], fd_deriv(f); atol = 1.0e-5)
    end

    @testset "pchip_interp — ForwardDiff.derivative MWE" begin
        deriv = ForwardDiff.derivative(
            t -> pchip_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap()), 1.0
        )
        fd = fd_deriv(t -> pchip_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap()))
        @test isapprox(deriv, fd; atol = 1.0e-5)
    end

    @testset "pchip_interp — interpolant (PreCompute + OnTheFly)" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        itp_pre = pchip_interp(d .* x_base, y_base; coeffs = PreCompute(), extrap = ExtendExtrap())
        itp_otf = pchip_interp(d .* x_base, y_base; coeffs = OnTheFly(), extrap = ExtendExtrap())
        v_pre = itp_pre(xq_scalar)
        v_otf = itp_otf(xq_scalar)
        @test v_pre isa ForwardDiff.Dual
        @test v_otf isa ForwardDiff.Dual
        # Both strategies should give same primal
        @test ForwardDiff.value(v_pre) ≈ ForwardDiff.value(v_otf)
    end

    @testset "pchip_interp — vector oneshot with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        result = pchip_interp(d .* x_base, y_base, xq_vec; extrap = ExtendExtrap())
        @test eltype(result) <: ForwardDiff.Dual
        ref = pchip_interp(x_base, y_base, xq_vec; extrap = ExtendExtrap())
        @test ForwardDiff.value.(result) ≈ ref
    end

    @testset "pchip_interp — adjoint with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        adj = pchip_adjoint(d .* x_base, y_base, xq_vec; extrap = ExtendExtrap())
        @test adj isa FI.PchipAdjoint1D
        y_bar = randn(length(xq_vec))
        f_bar = adj(y_bar)
        @test eltype(f_bar) <: ForwardDiff.Dual
    end

    @testset "pchip_interp — Range{Dual} grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x_range = d .* range(0.0, 2π, 30)
        y_r = sin.(range(0.0, 2π, 30))
        v = pchip_interp(x_range, y_r, xq_scalar; extrap = ExtendExtrap())
        @test v isa ForwardDiff.Dual
        @test ForwardDiff.value(v) ≈ pchip_interp(collect(range(0.0, 2π, 30)), y_r, xq_scalar; extrap = ExtendExtrap())
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           cardinal_interp                                              ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cardinal_interp — scalar oneshot with Dual grid" begin
        f = t -> cardinal_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap())
        v = f(ForwardDiff.Dual{:tag}(1.0, 1.0))
        @test v isa ForwardDiff.Dual
        @test ForwardDiff.value(v) ≈ cardinal_interp(x_base, y_base, xq_scalar; extrap = ExtendExtrap())
        @test isapprox(ForwardDiff.partials(v)[1], fd_deriv(f); atol = 1.0e-5)
    end

    @testset "cardinal_interp — ForwardDiff.derivative MWE" begin
        deriv = ForwardDiff.derivative(
            t -> cardinal_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap()), 1.0
        )
        fd = fd_deriv(t -> cardinal_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap()))
        @test isapprox(deriv, fd; atol = 1.0e-5)
    end

    @testset "cardinal_interp — interpolant (PreCompute + OnTheFly)" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        itp_pre = cardinal_interp(d .* x_base, y_base; coeffs = PreCompute(), extrap = ExtendExtrap())
        itp_otf = cardinal_interp(d .* x_base, y_base; coeffs = OnTheFly(), extrap = ExtendExtrap())
        v_pre = itp_pre(xq_scalar)
        v_otf = itp_otf(xq_scalar)
        @test v_pre isa ForwardDiff.Dual
        @test v_otf isa ForwardDiff.Dual
        @test ForwardDiff.value(v_pre) ≈ ForwardDiff.value(v_otf)
    end

    @testset "cardinal_interp — with tension parameter" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        v = cardinal_interp(d .* x_base, y_base, xq_scalar; tension = 0.5, extrap = ExtendExtrap())
        @test v isa ForwardDiff.Dual
        @test ForwardDiff.value(v) ≈ cardinal_interp(x_base, y_base, xq_scalar; tension = 0.5, extrap = ExtendExtrap())
    end

    @testset "cardinal_interp — adjoint with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        adj = cardinal_adjoint(d .* x_base, xq_vec; tension = 0.0, extrap = ExtendExtrap())
        @test adj isa FI.CardinalAdjoint1D
        y_bar = randn(length(xq_vec))
        f_bar = adj(y_bar)
        @test eltype(f_bar) <: ForwardDiff.Dual
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           akima_interp                                                 ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "akima_interp — scalar oneshot with Dual grid" begin
        f = t -> akima_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap())
        v = f(ForwardDiff.Dual{:tag}(1.0, 1.0))
        @test v isa ForwardDiff.Dual
        @test ForwardDiff.value(v) ≈ akima_interp(x_base, y_base, xq_scalar; extrap = ExtendExtrap())
        @test isapprox(ForwardDiff.partials(v)[1], fd_deriv(f); atol = 1.0e-5)
    end

    @testset "akima_interp — ForwardDiff.derivative MWE" begin
        deriv = ForwardDiff.derivative(
            t -> akima_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap()), 1.0
        )
        fd = fd_deriv(t -> akima_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap()))
        @test isapprox(deriv, fd; atol = 1.0e-5)
    end

    @testset "akima_interp — interpolant (PreCompute + OnTheFly)" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        itp_pre = akima_interp(d .* x_base, y_base; coeffs = PreCompute(), extrap = ExtendExtrap())
        itp_otf = akima_interp(d .* x_base, y_base; coeffs = OnTheFly(), extrap = ExtendExtrap())
        v_pre = itp_pre(xq_scalar)
        v_otf = itp_otf(xq_scalar)
        @test v_pre isa ForwardDiff.Dual
        @test v_otf isa ForwardDiff.Dual
        @test ForwardDiff.value(v_pre) ≈ ForwardDiff.value(v_otf)
    end

    @testset "akima_interp — adjoint with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        adj = akima_adjoint(d .* x_base, y_base, xq_vec; extrap = ExtendExtrap())
        @test adj isa FI.AkimaAdjoint1D
        y_bar = randn(length(xq_vec))
        f_bar = adj(y_bar)
        @test eltype(f_bar) <: ForwardDiff.Dual
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Akima adjoint coverage: small grids + NoExtrap               ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "akima_adjoint — n=2 grid (Dual)" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x2 = d .* [1.0, 2.0]
        y2 = [3.0, 5.0]
        adj = akima_adjoint(x2, y2, [1.5]; extrap = ExtendExtrap())
        f_bar = adj(randn(1))
        @test length(f_bar) == 2
        @test eltype(f_bar) <: ForwardDiff.Dual
    end

    @testset "akima_adjoint — n=3 grid (Dual)" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x3 = d .* [1.0, 2.0, 3.0]
        y3 = [1.0, 4.0, 2.0]
        adj = akima_adjoint(x3, y3, [1.5, 2.5]; extrap = ExtendExtrap())
        f_bar = adj(randn(2))
        @test length(f_bar) == 3
        @test eltype(f_bar) <: ForwardDiff.Dual
    end

    @testset "akima_adjoint — wsum=0 branch (constant data, Dual grid)" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        # Constant y → all secants equal → wsum=0 in Akima weighted average
        x_flat = d .* collect(1.0:1.0:6.0)
        y_flat = fill(5.0, 6)
        adj = akima_adjoint(x_flat, y_flat, [2.5, 3.5, 4.5]; extrap = ExtendExtrap())
        f_bar = adj(randn(3))
        @test length(f_bar) == 6
        @test eltype(f_bar) <: ForwardDiff.Dual
    end

    @testset "akima_adjoint — NoExtrap domain check (Dual grid)" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x_dual = d .* collect(1.0:1.0:5.0)
        y_data = [1.0, 2.0, 1.5, 3.0, 2.0]
        # In-domain queries — should not throw
        adj = akima_adjoint(x_dual, y_data, [1.0, 3.0, 5.0]; extrap = NoExtrap())
        @test adj isa FI.AkimaAdjoint1D
        # Out-of-domain query — should throw
        @test_throws DomainError akima_adjoint(x_dual, y_data, [0.5]; extrap = NoExtrap())
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Regression: Float path through same API                      ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Regression: Float scalar path zero-alloc" begin
        x = collect(range(0.0, 2π, 50))
        y = sin.(x)

        itp_p = pchip_interp(x, y)
        itp_c = cardinal_interp(x, y)
        itp_a = akima_interp(x, y)

        # warmup
        itp_p(2.5); itp_c(2.5); itp_a(2.5)
        @test (@allocated itp_p(2.5)) <= ALLOC_THRESHOLD
        @test (@allocated itp_c(2.5)) <= ALLOC_THRESHOLD
        @test (@allocated itp_a(2.5)) <= ALLOC_THRESHOLD
    end
end
