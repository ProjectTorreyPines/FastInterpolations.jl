# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║     Constant + Quadratic — Dual Grid Tests (ForwardDiff)                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

using Test
using FastInterpolations
using ForwardDiff
using LinearAlgebra

const FI = FastInterpolations

@testset "Constant + Quadratic — Dual Grid" begin

    x_base = collect(range(0.0, 5.0, 20))
    y_base = sin.(x_base)
    xq_scalar = 2.5
    xq_vec = [0.5, 1.5, 2.5, 3.5, 4.5]
    h_fd = 1.0e-7
    fd_deriv(f) = (f(1.0 + h_fd) - f(1.0 - h_fd)) / (2h_fd)

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           constant_interp                                              ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "constant_interp — scalar oneshot with Dual grid" begin
        # Constant interpolation returns y[idx] directly — no grid arithmetic.
        # Result is Float (y type), not Dual (grid type). This is correct:
        # ∂/∂(grid_param) of constant_interp = 0 almost everywhere.
        f = t -> constant_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap())
        v = f(ForwardDiff.Dual{:tag}(1.0, 1.0))
        # Constant kernel returns y[idx] which is Float — no grid partials propagate
        @test ForwardDiff.value(v) ≈ constant_interp(x_base, y_base, xq_scalar; extrap = ExtendExtrap())
    end

    @testset "constant_interp — interpolant with Dual grid constructs" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        itp = constant_interp(d .* x_base, y_base; extrap = ExtendExtrap())
        v = itp(xq_scalar)
        itp_f = constant_interp(x_base, y_base; extrap = ExtendExtrap())
        @test ForwardDiff.value(v) ≈ itp_f(xq_scalar)
    end

    @testset "constant_interp — vector oneshot with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        result = constant_interp(d .* x_base, y_base, xq_vec; extrap = ExtendExtrap())
        ref = constant_interp(x_base, y_base, xq_vec; extrap = ExtendExtrap())
        @test ForwardDiff.value.(result) ≈ ref
    end

    @testset "constant_interp — Range{Dual} grid (constructs)" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        # Range{Dual} construction works; value may differ from Vector path
        # because DirectSearch index uses primal-based trunc (rounding differences)
        v = constant_interp(d .* range(0.0, 5.0, 20), y_base, xq_scalar; extrap = ExtendExtrap())
        @test isfinite(ForwardDiff.value(v))
    end

    @testset "constant_interp — adjoint with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        adj = constant_adjoint(d .* x_base, xq_vec; extrap = ExtendExtrap())
        y_bar = randn(length(xq_vec))
        f_bar = adj(y_bar)
        @test length(f_bar) == length(x_base)
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           quadratic_interp                                             ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "quadratic_interp — scalar oneshot with Dual grid" begin
        f = t -> quadratic_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap())
        v = f(ForwardDiff.Dual{:tag}(1.0, 1.0))
        @test v isa ForwardDiff.Dual
        @test ForwardDiff.value(v) ≈ quadratic_interp(x_base, y_base, xq_scalar; extrap = ExtendExtrap())
        @test isapprox(ForwardDiff.partials(v)[1], fd_deriv(f); atol = 1.0e-5)
    end

    @testset "quadratic_interp — ForwardDiff.derivative MWE" begin
        deriv = ForwardDiff.derivative(
            t -> quadratic_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap()), 1.0
        )
        fd = fd_deriv(t -> quadratic_interp(t .* x_base, y_base, xq_scalar; extrap = ExtendExtrap()))
        @test isapprox(deriv, fd; atol = 1.0e-5)
    end

    @testset "quadratic_interp — interpolant with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        itp = quadratic_interp(d .* x_base, y_base; extrap = ExtendExtrap())
        v = itp(xq_scalar)
        @test v isa ForwardDiff.Dual
        itp_f = quadratic_interp(x_base, y_base; extrap = ExtendExtrap())
        @test ForwardDiff.value(v) ≈ itp_f(xq_scalar)
    end

    @testset "quadratic_interp — vector oneshot with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        result = quadratic_interp(d .* x_base, y_base, xq_vec; extrap = ExtendExtrap())
        @test eltype(result) <: ForwardDiff.Dual
        ref = quadratic_interp(x_base, y_base, xq_vec; extrap = ExtendExtrap())
        @test ForwardDiff.value.(result) ≈ ref
    end

    @testset "quadratic_interp — Range{Dual} grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        v = quadratic_interp(d .* range(0.0, 5.0, 20), y_base, xq_scalar; extrap = ExtendExtrap())
        @test v isa ForwardDiff.Dual
    end

    @testset "quadratic_interp — adjoint with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        adj = quadratic_adjoint(d .* x_base, xq_vec; extrap = ExtendExtrap())
        y_bar = randn(length(xq_vec))
        f_bar = adj(y_bar)
        @test eltype(f_bar) <: ForwardDiff.Dual
    end

    @testset "quadratic_interp — BC variants with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x_dual = d .* x_base
        for bc in [Left(QuadraticFit()), Right(Deriv2(0.0)), Left(Deriv1(0.0))]
            v = quadratic_interp(x_dual, y_base, xq_scalar; bc = bc, extrap = ExtendExtrap())
            @test v isa ForwardDiff.Dual
        end
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           ND paths                                                     ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "constant_interp ND — Dual grid 2D" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        xv = collect(1.0:0.5:5.0)
        yv = collect(2.0:0.5:6.0)
        data = [sin(xi) * cos(yi) for xi in xv, yi in yv]

        # Constant ND: construction works, result matches Float (no grid arithmetic)
        itp = constant_interp((d .* xv, d .* yv), data; extrap = ExtendExtrap())
        v = itp((2.55, 3.55))
        itp_f = constant_interp((xv, yv), data)
        @test ForwardDiff.value(v) ≈ itp_f((2.55, 3.55))
    end

    # Note: Quadratic ND with Dual grids requires deeper changes in quadratic_nd_build.jl
    # (coefficient precomputation per-axis) — deferred to follow-up PR with Cubic ND.

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Extrapolation modes + edge cases with Dual grids             ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Extrap modes: NoExtrap, ClampExtrap with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x_dual = d .* x_base
        # NoExtrap — endpoint queries should be in-domain (primal-based check)
        v = quadratic_interp(x_dual, y_base, first(x_base); extrap = NoExtrap())
        @test ForwardDiff.value(v) ≈ quadratic_interp(x_base, y_base, first(x_base))
        # ClampExtrap — OOB query clamped to boundary
        v_clamp = quadratic_interp(x_dual, y_base, -1.0; extrap = ClampExtrap())
        @test isfinite(ForwardDiff.value(v_clamp))
    end

    @testset "Quadratic 2-point grid with Dual" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        x2 = d .* [1.0, 3.0]
        y2 = [2.0, 6.0]
        v = quadratic_interp(x2, y2, 2.0; bc = Left(Deriv1(0.0)), extrap = ExtendExtrap())
        @test v isa ForwardDiff.Dual
    end

    @testset "Quadratic derivative with Dual grid" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        v1 = quadratic_interp(d .* x_base, y_base, xq_scalar; deriv = DerivOp(1), extrap = ExtendExtrap())
        @test v1 isa ForwardDiff.Dual
        v2 = quadratic_interp(d .* x_base, y_base, xq_scalar; deriv = DerivOp(2), extrap = ExtendExtrap())
        @test v2 isa ForwardDiff.Dual
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║           Regression: Float path                                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Regression: Float scalar path zero-alloc" begin
        x = collect(range(0.0, 5.0, 50))
        y = sin.(x)

        itp_c = constant_interp(x, y)
        itp_q = quadratic_interp(x, y)

        itp_c(2.5); itp_q(2.5)
        @test (@allocated itp_c(2.5)) <= ALLOC_THRESHOLD
        @test (@allocated itp_q(2.5)) <= ALLOC_THRESHOLD
    end

    # ========================================================================
    # PeriodicBC + Dual grid — verifies `_PromotableValue` guard in
    # `_extend_exclusive` / `_periodic_extend_1d_pooled!` passes Dual eltype
    # through unchanged (Dual is not `<: _PromotableValue`).
    # ========================================================================
    @testset "Constant PeriodicBC — Dual grid preserves type" begin
        # Constant interp returns `y[idx]` directly — the Dual grid only affects
        # which index is picked, not the returned value's type (no arithmetic
        # with xq). The invariant being tested here is that the grid stores
        # Dual (unchanged by the periodic extension path) and that the call
        # succeeds without promoting/stripping the grid's Dual eltype.
        x_dual = [ForwardDiff.Dual{:pbc}(Float64(i), 0.0) for i in 0:5]
        y = Float64[0, 1, 2, 3, 4, 5]

        itp = constant_interp(x_dual, y; bc = PeriodicBC(endpoint = :exclusive, period = 6.0))
        @test eltype(itp.x) <: ForwardDiff.Dual
        @test length(itp.x) == 7

        q = ForwardDiff.Dual{:pbc}(2.5, 1.0)
        out = constant_interp(x_dual, y, q; bc = PeriodicBC(endpoint = :exclusive, period = 6.0))
        @test out == 2.0 || out == 3.0  # one of the neighboring y-values
    end
end
