# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║          ND — ForwardDiff.Dual grid support (OnTheFly + PreCompute)       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Tests for ND interpolation with ForwardDiff.Dual grid elements.
# Phase 1: OnTheFly path (constraint lift in hetero_oneshot.jl)
# Phase 2: PreCompute path (partials pipeline type widening)
#
# Run standalone:
#     cc-julia-test-runner . ext/test_nd_dual_grid.jl

using Test
using FastInterpolations
using ForwardDiff

const FI = FastInterpolations

# ── Shared test data ──────────────────────────────────────────────────────

const xv_base = collect(range(0.0, 5.0, 15))
const yv_base = collect(range(0.0, 4.0, 12))
const data_2d = [sin(xi) * cos(yi) for xi in xv_base, yi in yv_base]
const q_2d = (2.3, 1.7)
const h_fd = 1.0e-7

# FD helper: central difference for scalar t
fd_deriv(f) = (f(1.0 + h_fd) - f(1.0 - h_fd)) / (2h_fd)

# FD helper: central difference gradient for vector t
function fd_gradient(f, t0::Vector{Float64})
    g = similar(t0)
    for i in eachindex(t0)
        tp = copy(t0); tp[i] += h_fd
        tm = copy(t0); tm[i] -= h_fd
        g[i] = (f(tp) - f(tm)) / (2h_fd)
    end
    return g
end

@testset "ND Dual Grid" begin

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Phase 1: OnTheFly — Homogeneous methods (coeffs=OnTheFly())        ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Cubic ND OnTheFly — all-Dual grids" begin
        f_interp = t -> cubic_interp(
            (t .* xv_base, t .* yv_base), data_2d, q_2d;
            coeffs = OnTheFly(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "Quadratic ND OnTheFly — all-Dual grids" begin
        f_interp = t -> quadratic_interp(
            (t .* xv_base, t .* yv_base), data_2d, q_2d;
            coeffs = OnTheFly(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "Hetero (Cubic × Linear) OnTheFly — all-Dual grids" begin
        f_interp = t -> interp(
            (t .* xv_base, t .* yv_base), data_2d, q_2d;
            method = (CubicInterp(), LinearInterp()),
            extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "PCHIP ND OnTheFly — all-Dual grids" begin
        f_interp = t -> interp(
            (t .* xv_base, t .* yv_base), data_2d, q_2d;
            method = PchipInterp(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "Cardinal ND OnTheFly — all-Dual grids" begin
        f_interp = t -> interp(
            (t .* xv_base, t .* yv_base), data_2d, q_2d;
            method = CardinalInterp(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "Akima ND OnTheFly — all-Dual grids" begin
        f_interp = t -> interp(
            (t .* xv_base, t .* yv_base), data_2d, q_2d;
            method = AkimaInterp(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Phase 1: Hybrid grids — per-axis Dual                              ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Cubic ND OnTheFly — axis-1 Dual only" begin
        # Only axis-1 grid is Dual → partial wrt t should reflect axis-1 sensitivity
        f_interp = t -> cubic_interp(
            (t .* xv_base, yv_base), data_2d, q_2d;
            coeffs = OnTheFly(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "Cubic ND OnTheFly — axis-2 Dual only" begin
        f_interp = t -> cubic_interp(
            (xv_base, t .* yv_base), data_2d, q_2d;
            coeffs = OnTheFly(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "Cubic ND OnTheFly — per-axis ForwardDiff.gradient" begin
        # Two-variable gradient: t = [t1, t2], grids = (t1*x, t2*y)
        f_grad = t -> cubic_interp(
            (t[1] .* xv_base, t[2] .* yv_base), data_2d, q_2d;
            coeffs = OnTheFly(), extrap = ExtendExtrap(),
        )
        ad_grad = ForwardDiff.gradient(f_grad, [1.0, 1.0])
        fd_grad = fd_gradient(f_grad, [1.0, 1.0])
        @test ad_grad ≈ fd_grad rtol = 1.0e-5
        # Verify both axes contribute (non-zero partials)
        @test abs(ad_grad[1]) > 1.0e-10
        @test abs(ad_grad[2]) > 1.0e-10
    end

    @testset "PCHIP ND OnTheFly — per-axis gradient" begin
        f_grad = t -> interp(
            (t[1] .* xv_base, t[2] .* yv_base), data_2d, q_2d;
            method = PchipInterp(), extrap = ExtendExtrap(),
        )
        ad_grad = ForwardDiff.gradient(f_grad, [1.0, 1.0])
        fd_grad = fd_gradient(f_grad, [1.0, 1.0])
        @test ad_grad ≈ fd_grad rtol = 1.0e-5
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Phase 1: Hybrid grids — Range{Dual} + Vector{Float}               ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Cubic ND OnTheFly — Range{Dual} × Vector{Float}" begin
        xr = range(0.0, 5.0, 15)
        f_interp = t -> cubic_interp(
            (t .* xr, yv_base), data_2d, q_2d;
            coeffs = OnTheFly(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "PCHIP ND OnTheFly — Range{Dual} × Vector{Float}" begin
        xr = range(0.0, 5.0, 15)
        f_interp = t -> interp(
            (t .* xr, yv_base), data_2d, q_2d;
            method = PchipInterp(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "Hetero (Cubic × Linear) — Range{Dual} × Vector{Dual}" begin
        xr = range(0.0, 5.0, 15)
        f_interp = t -> interp(
            (t .* xr, t .* yv_base), data_2d, q_2d;
            method = (CubicInterp(), LinearInterp()),
            extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Phase 1: Primal correctness — Dual values match Float path         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Cubic ND OnTheFly — Dual primal matches Float" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        v_dual = cubic_interp(
            (d .* xv_base, d .* yv_base), data_2d, q_2d;
            coeffs = OnTheFly(), extrap = ExtendExtrap(),
        )
        v_float = cubic_interp(
            (xv_base, yv_base), data_2d, q_2d;
            coeffs = OnTheFly(), extrap = ExtendExtrap(),
        )
        @test ForwardDiff.value(v_dual) ≈ v_float
    end

    @testset "PCHIP ND — Dual primal matches Float" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        v_dual = interp(
            (d .* xv_base, d .* yv_base), data_2d, q_2d;
            method = PchipInterp(), extrap = ExtendExtrap(),
        )
        v_float = interp(
            (xv_base, yv_base), data_2d, q_2d;
            method = PchipInterp(), extrap = ExtendExtrap(),
        )
        @test ForwardDiff.value(v_dual) ≈ v_float
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Phase 1: Batch queries                                             ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Cubic ND OnTheFly batch — all-Dual primals match Float" begin
        queries = [(2.3, 1.7), (1.5, 2.5), (3.1, 0.8)]
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        # Per-query scalar OnTheFly to verify primals
        for q in queries
            v_dual = cubic_interp(
                (d .* xv_base, d .* yv_base), data_2d, q;
                coeffs = OnTheFly(), extrap = ExtendExtrap(),
            )
            v_float = cubic_interp(
                (xv_base, yv_base), data_2d, q;
                coeffs = OnTheFly(), extrap = ExtendExtrap(),
            )
            @test ForwardDiff.value(v_dual) ≈ v_float
        end
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Phase 2: PreCompute — Homogeneous one-shot                         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Cubic ND PreCompute — all-Dual grids" begin
        f_interp = t -> cubic_interp(
            (t .* xv_base, t .* yv_base), data_2d, q_2d;
            coeffs = PreCompute(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "Quadratic ND PreCompute — all-Dual grids" begin
        f_interp = t -> quadratic_interp(
            (t .* xv_base, t .* yv_base), data_2d, q_2d;
            coeffs = PreCompute(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "Cubic ND PreCompute — per-axis gradient" begin
        f_grad = t -> cubic_interp(
            (t[1] .* xv_base, t[2] .* yv_base), data_2d, q_2d;
            coeffs = PreCompute(), extrap = ExtendExtrap(),
        )
        ad_grad = ForwardDiff.gradient(f_grad, [1.0, 1.0])
        fd_grad = fd_gradient(f_grad, [1.0, 1.0])
        @test ad_grad ≈ fd_grad rtol = 1.0e-5
    end

    @testset "Cubic ND PreCompute — axis-1 Dual only" begin
        f_interp = t -> cubic_interp(
            (t .* xv_base, yv_base), data_2d, q_2d;
            coeffs = PreCompute(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Phase 2: PreCompute — Heterogeneous one-shot                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Hetero (Cubic × Linear) PreCompute — all-Dual grids" begin
        f_interp = t -> interp(
            (t .* xv_base, t .* yv_base), data_2d, q_2d;
            method = (CubicInterp(), LinearInterp()),
            coeffs = PreCompute(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    @testset "Hetero PreCompute batch — all-Dual grids" begin
        queries = [(2.3, 1.7), (1.5, 2.5), (3.1, 0.8)]
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        result = interp(
            (d .* xv_base, d .* yv_base), data_2d, queries;
            method = (CubicInterp(), LinearInterp()),
            extrap = ExtendExtrap(),
        )
        ref = interp(
            (xv_base, yv_base), data_2d, queries;
            method = (CubicInterp(), LinearInterp()),
            extrap = ExtendExtrap(),
        )
        @test ForwardDiff.value.(result) ≈ ref
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Phase 2: PreCompute — Interpolant (2-arg persistent callable)      ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Cubic ND Interpolant — Dual grid construct + eval" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        itp = cubic_interp((d .* xv_base, d .* yv_base), data_2d; extrap = ExtendExtrap())
        v = itp(q_2d)
        itp_f = cubic_interp((xv_base, yv_base), data_2d; extrap = ExtendExtrap())
        @test ForwardDiff.value(v) ≈ itp_f(q_2d)
    end

    @testset "Quadratic ND Interpolant — Dual grid construct + eval" begin
        d = ForwardDiff.Dual{:tag}(1.0, 1.0)
        itp = quadratic_interp((d .* xv_base, d .* yv_base), data_2d; extrap = ExtendExtrap())
        v = itp(q_2d)
        itp_f = quadratic_interp((xv_base, yv_base), data_2d; extrap = ExtendExtrap())
        @test ForwardDiff.value(v) ≈ itp_f(q_2d)
    end

    @testset "Cubic ND Interpolant — ForwardDiff.derivative" begin
        f_interp = t -> begin
            itp = cubic_interp((t .* xv_base, t .* yv_base), data_2d; extrap = ExtendExtrap())
            itp(q_2d)
        end
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Phase 2: PreCompute — Range{Dual} hybrid                          ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Cubic ND PreCompute — Range{Dual} × Vector{Float}" begin
        xr = range(0.0, 5.0, 15)
        f_interp = t -> cubic_interp(
            (t .* xr, yv_base), data_2d, q_2d;
            coeffs = PreCompute(), extrap = ExtendExtrap(),
        )
        ad_val = ForwardDiff.derivative(f_interp, 1.0)
        fd_val = fd_deriv(f_interp)
        @test ad_val ≈ fd_val rtol = 1.0e-5
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Phase 2: Float regression for PreCompute path                      ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Float regression — Cubic ND PreCompute still works" begin
        v = cubic_interp(
            (xv_base, yv_base), data_2d, q_2d;
            coeffs = PreCompute(), extrap = ExtendExtrap(),
        )
        @test v isa Float64
        @test isfinite(v)
    end

    @testset "Float regression — Quadratic ND PreCompute still works" begin
        v = quadratic_interp(
            (xv_base, yv_base), data_2d, q_2d;
            coeffs = PreCompute(), extrap = ExtendExtrap(),
        )
        @test v isa Float64
        @test isfinite(v)
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Regression: Float grid ND path unchanged                           ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "Float regression — Cubic ND OnTheFly still works" begin
        v = cubic_interp(
            (xv_base, yv_base), data_2d, q_2d;
            coeffs = OnTheFly(), extrap = ExtendExtrap(),
        )
        @test v isa Float64
        @test isfinite(v)
    end

    @testset "Float regression — interp() hetero still works" begin
        v = interp(
            (xv_base, yv_base), data_2d, q_2d;
            method = (CubicInterp(), LinearInterp()),
            extrap = ExtendExtrap(),
        )
        @test v isa Float64
        @test isfinite(v)
    end

end
