# Duck-typing tests for non-promotable Tv (e.g., SVector) crossed with non-
# promotable Tq (e.g., ForwardDiff.Dual).
#
# These cases hit `_output_eltype(Tv, Tg, Tq)`'s `Base.promote_op` fallback —
# `promote_type` can't model `SVector × Dual` and similar third-party chains.
# Tests pin the expected return types and container concreteness for both
# the persistent callable path and the one-shot path, plus type-stability
# (`@inferred`) and raw-eltype non-regression contracts.

@testitem "Persistent path — duck Tv × Tq widens via build + call" begin
    using StaticArrays, ForwardDiff

    x = collect(1.0:10.0)
    y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    dy_sv = [SA[1.0, 2.0, 3.0] for _ in 1:10]
    xq_d = ForwardDiff.Dual{Nothing}(2.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "scalar Dual" begin
        @test linear_interp(x, y_sv)(xq_d) isa SVector{3, D}
        @test cubic_interp(x, y_sv)(xq_d) isa SVector{3, D}
        @test quadratic_interp(x, y_sv)(xq_d) isa SVector{3, D}
        @test hermite_interp(x, y_sv, dy_sv)(xq_d) isa SVector{3, D}
        @test constant_interp(x, y_sv)(xq_d) isa SVector{3, D}
    end

    @testset "batch Vector{Dual}" begin
        @test linear_interp(x, y_sv)(xq_dv) isa Vector{SVector{3, D}}
        @test cubic_interp(x, y_sv)(xq_dv) isa Vector{SVector{3, D}}
        @test quadratic_interp(x, y_sv)(xq_dv) isa Vector{SVector{3, D}}
        @test hermite_interp(x, y_sv, dy_sv)(xq_dv) isa Vector{SVector{3, D}}
        @test constant_interp(x, y_sv)(xq_dv) isa Vector{SVector{3, D}}
    end

    @testset "ND batch — SVector data × Tuple{Dual, Dual}" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data_sv = [SA[Float64(i + j), 2.0(i + j), 3.0(i + j)] for i in 1:5, j in 1:5]
        q_dv = [(ForwardDiff.Dual{Nothing}(2.5 + 0.1i, 1.0),
                 ForwardDiff.Dual{Nothing}(3.5 + 0.1i, 0.0)) for i in 1:5]
        @test linear_interp((xg, yg), data_sv)(q_dv) isa Vector{SVector{3, D}}
        @test cubic_interp((xg, yg), data_sv)(q_dv) isa Vector{SVector{3, D}}
        @test constant_interp((xg, yg), data_sv)(q_dv) isa Vector{SVector{3, D}}
    end

    @testset "ForwardDiff.derivative through indexed component (Issue #144 MWE)" begin
        for builder in (
                () -> linear_interp(x, y_sv),
                () -> cubic_interp(x, y_sv),
                () -> quadratic_interp(x, y_sv),
                () -> hermite_interp(x, y_sv, dy_sv),
                () -> constant_interp(x, y_sv),
            )
            itp = builder()
            d = ForwardDiff.derivative(t -> itp(t)[1], 2.5)
            @test d isa Float64
            @test isfinite(d)
        end
    end

    @testset "Plain Float64 y + Vector{Dual} batch (non-regression)" begin
        y = sin.(x)
        @test linear_interp(x, y)(xq_dv) isa Vector{D}
        @test cubic_interp(x, y)(xq_dv) isa Vector{D}
        @test quadratic_interp(x, y)(xq_dv) isa Vector{D}
        @test hermite_interp(x, y, cos.(x))(xq_dv) isa Vector{D}
        @test constant_interp(x, y)(xq_dv) isa Vector{D}
    end
end

@testitem "Oneshot path — duck Tv × Tq widens identically" begin
    using StaticArrays, ForwardDiff

    x = collect(1.0:10.0)
    y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    dy_sv = [SA[1.0, 2.0, 3.0] for _ in 1:10]
    xq_d = ForwardDiff.Dual{Nothing}(2.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "1D scalar 3-arg" begin
        @test linear_interp(x, y_sv, xq_d) isa SVector{3, D}
        @test cubic_interp(x, y_sv, xq_d) isa SVector{3, D}
        @test quadratic_interp(x, y_sv, xq_d) isa SVector{3, D}
        @test hermite_interp(x, y_sv, dy_sv, xq_d) isa SVector{3, D}
        @test constant_interp(x, y_sv, xq_d) isa SVector{3, D}
    end

    @testset "1D batch 3-arg (the 4 cases broken before this fix)" begin
        @test linear_interp(x, y_sv, xq_dv) isa Vector{SVector{3, D}}
        @test cubic_interp(x, y_sv, xq_dv) isa Vector{SVector{3, D}}
        @test quadratic_interp(x, y_sv, xq_dv) isa Vector{SVector{3, D}}
        @test hermite_interp(x, y_sv, dy_sv, xq_dv) isa Vector{SVector{3, D}}
        @test constant_interp(x, y_sv, xq_dv) isa Vector{SVector{3, D}}
    end

    @testset "ND batch 3-arg" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data_sv = [SA[Float64(i + j), 2.0(i + j), 3.0(i + j)] for i in 1:5, j in 1:5]
        q_dv = [(ForwardDiff.Dual{Nothing}(2.5 + 0.1i, 1.0),
                 ForwardDiff.Dual{Nothing}(3.5 + 0.1i, 0.0)) for i in 1:5]
        @test linear_interp((xg, yg), data_sv, q_dv) isa Vector{SVector{3, D}}
        @test cubic_interp((xg, yg), data_sv, q_dv) isa Vector{SVector{3, D}}
        @test quadratic_interp((xg, yg), data_sv, q_dv) isa Vector{SVector{3, D}}
        # Constant ND oneshot batch — fixed in acfc95acf (was `::Vector{Tv}` typeassert).
        @test constant_interp((xg, yg), data_sv, q_dv) isa Vector{SVector{3, D}}
    end
end

@testitem "Series path — duck-Tv × duck-Tq carrier propagates" begin
    using StaticArrays, ForwardDiff
    using FastInterpolations: Series

    x = collect(1.0:10.0)
    y_sv1 = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    y_sv2 = [SA[-1.0 * i, 0.5i, 2.5i] for i in 1:10]
    s_sv  = Series(y_sv1, y_sv2)
    xq_d  = ForwardDiff.Dual{Nothing}(5.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "scalar Dual xq, SVector y per series" begin
        # Returns Vector{T_out} of length K = number of series.
        @test linear_interp(x, s_sv, xq_d) isa Vector{SVector{3, D}}
        @test cubic_interp(x, s_sv, xq_d) isa Vector{SVector{3, D}}
        @test quadratic_interp(x, s_sv, xq_d) isa Vector{SVector{3, D}}
        @test constant_interp(x, s_sv, xq_d) isa Vector{SVector{3, D}}
    end

    @testset "batch Vector{Dual} xq, SVector y per series" begin
        # Returns Vector{Vector{T_out}}, one inner vector per series.
        @test linear_interp(x, s_sv, xq_dv) isa Vector{Vector{SVector{3, D}}}
        @test cubic_interp(x, s_sv, xq_dv) isa Vector{Vector{SVector{3, D}}}
        @test quadratic_interp(x, s_sv, xq_dv) isa Vector{Vector{SVector{3, D}}}
        @test constant_interp(x, s_sv, xq_dv) isa Vector{Vector{SVector{3, D}}}
    end
end

@testitem "Adjoint path — duck-Tv y_bar widens correctly" begin
    using StaticArrays, ForwardDiff

    x = collect(1.0:10.0)
    y_for_slope = sin.(x)
    xq_for_adj = [2.5, 5.5, 8.5]
    y_bar_sv = [SA[0.1, 0.2, 0.3], SA[0.4, 0.5, 0.6], SA[0.7, 0.8, 0.9]]
    y_bar_dv = [ForwardDiff.Dual{Nothing}(0.1 + i, 1.0) for i in 1:3]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "1D — slope-free adjoints (Linear/Constant/Cubic/Quadratic)" begin
        for fn in (linear_adjoint, constant_adjoint, cubic_adjoint, quadratic_adjoint)
            @test fn(x, xq_for_adj)(y_bar_sv) isa Vector{SVector{3, Float64}}
            @test fn(x, xq_for_adj)(y_bar_dv) isa Vector{D}
        end
    end

    @testset "1D — Hermite-family adjoints" begin
        for fn in (hermite_adjoint, cardinal_adjoint)
            @test fn(x, xq_for_adj)(y_bar_sv) isa Vector{SVector{3, Float64}}
            @test fn(x, xq_for_adj)(y_bar_dv) isa Vector{D}
        end
        # Pchip/Akima carry y in the constructor (slope is data-dependent).
        for fn in (pchip_adjoint, akima_adjoint)
            @test fn(x, y_for_slope, xq_for_adj)(y_bar_sv) isa Vector{SVector{3, Float64}}
            @test fn(x, y_for_slope, xq_for_adj)(y_bar_dv) isa Vector{D}
        end
    end

    @testset "ND adjoints (Linear/Constant/Cubic/Quadratic)" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        xq_nd = [(2.5, 3.5), (3.0, 4.0), (4.0, 2.5)]
        for fn in (linear_adjoint, constant_adjoint, cubic_adjoint, quadratic_adjoint)
            @test fn((xg, yg), xq_nd)(y_bar_sv) isa Matrix{SVector{3, Float64}}
            @test fn((xg, yg), xq_nd)(y_bar_dv) isa Matrix{D}
        end
    end
end

@testitem "Type stability + raw-eltype contracts" begin
    using Test, StaticArrays, ForwardDiff
    using FastInterpolations: _output_eltype

    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "Constant scalar @inferred — no Union{Tv, Tq} return" begin
        # 1D: Int y, Float/Int xq — the kernel's `* one(dL)` and the right-edge
        # short-circuit must agree on the return type.
        x = 1:10
        y = (1:10).^2
        itp = constant_interp(x, y)
        @test (@inferred itp(1.0)) === 1.0     # at first(x)
        @test (@inferred itp(10.0)) === 100.0  # at last(x): the short-circuit branch
        @test (@inferred itp(5.5)) === 25.0    # interior
        @test (@inferred itp(1)) === 1         # Int xq stays Int
        @test (@inferred itp(10)) === 100
        @test Base.return_types(itp, (Float64,)) == [Float64]
        @test Base.return_types(itp, (Int,)) == [Int]

        # ND: same contract through the anchor short-circuit path
        xg = 1:5; yg = 1:5
        data = [i + j for i in 1:5, j in 1:5]
        itp_nd = constant_interp((xg, yg), data)
        @test (@inferred itp_nd((1.0, 1.0))) isa Float64
        @test (@inferred itp_nd((1, 1))) isa Int
        @test Base.return_types(itp_nd, (Tuple{Float64, Float64},)) == [Float64]
        @test Base.return_types(itp_nd, (Tuple{Int, Int},)) == [Int]

        # Dual xq through Float y
        itp_f = constant_interp(1.0:10.0, collect(1.0:10.0))
        @test (@inferred itp_f(ForwardDiff.Dual{Nothing}(5.5, 1.0))) isa D
    end

    @testset "Constant natural-promote — scalar Int×Int → Int; batch widens" begin
        # Scalar: kernel direct, fully-Int chain stays Int (kernel returns
        # `y[idx] * one(Int) = Int`).
        x = collect(1:10)
        y = collect(10:10:100)
        @test constant_interp(x, y)(3) === 30
        # Batch: trait applies Int→Float upgrade (same machinery as every
        # other method — no Constant-specific helper).
        @test constant_interp(x, y)([2, 5, 8]) isa Vector{Float64}
    end

    @testset "_output_eltype trait — type table" begin
        @test _output_eltype(Float64, Float64, Float64) === Float64
        @test _output_eltype(Int, Int, Int) === Float64                       # Int → Float upgrade
        @test _output_eltype(Int, Int, Float64) === Float64
        @test _output_eltype(Float32, Float64, Float64) === Float64
        @test _output_eltype(SVector{3, Float64}, Float64, Float64) === SVector{3, Float64}
        @test _output_eltype(SVector{3, Float64}, Float64, D) === SVector{3, D}         # Issue #144
        @test _output_eltype(SVector{3, Int}, Float64, Float64) === SVector{3, Float64} # silent Int-truncation fix
        @test _output_eltype(SVector{3, Int}, Float64, D) === SVector{3, D}
        @test _output_eltype(Complex{Int}, Float64, Float64) === ComplexF64
        @test _output_eltype(D, Float64, Float64) === D

        # Duck-type fallback: when `*` is undefined, return Tv unchanged.
        struct _DuckNoOp end
        @test _output_eltype(_DuckNoOp, Float64, Float64) === _DuckNoOp
    end
end

# Advanced AD shapes: Dual x grid, grid-offset sensitivity, nested Dual,
# ForwardDiff.gradient + Hessian. These pin cases beyond the basic
# {scalar, batch} × duck-Tv × duck-Tq matrix already covered above.
@testitem "Advanced AD — Dual grid, grid-offset, nested, gradient" begin
    using StaticArrays, ForwardDiff

    x_f = collect(1.0:10.0)
    y_f = sin.(x_f); dy_f = cos.(x_f)
    y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    dy_sv = [SA[1.0, 2.0, 3.0] for _ in 1:10]
    x_dg = [ForwardDiff.Dual{Nothing}(Float64(i), 1.0) for i in 1:10]
    xq_f = 5.5
    xq_d = ForwardDiff.Dual{Nothing}(5.5, 1.0)
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "Dual x grid carrier flows through Tv × Tq" begin
        # Persistent + one-shot, scalar + batch, both Float y and SVector y.
        for fn in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            @test fn(x_dg, y_f)(xq_f) isa D
            @test fn(x_dg, y_f)(xq_d) isa D
            @test fn(x_dg, y_sv)(xq_f) isa SVector{3, D}
            @test fn(x_dg, y_sv)(xq_d) isa SVector{3, D}
            # One-shot mirror
            @test fn(x_dg, y_f, xq_d) isa D
            @test fn(x_dg, y_sv, xq_d) isa SVector{3, D}
        end
        @test hermite_interp(x_dg, y_f, dy_f)(xq_d) isa D
        @test hermite_interp(x_dg, y_sv, dy_sv)(xq_d) isa SVector{3, D}
    end

    @testset "ForwardDiff.derivative w.r.t. grid offset" begin
        # `∂/∂δ fn(x .+ δ, y, xq)` — grid sensitivity via AD. Persistent + one-shot.
        for fn in (linear_interp, cubic_interp, constant_interp)
            d_pers = ForwardDiff.derivative(δ -> fn(x_f .+ δ, y_f)(xq_f), 0.0)
            d_one  = ForwardDiff.derivative(δ -> fn(x_f .+ δ, y_f, xq_f), 0.0)
            @test d_pers isa Float64 && isfinite(d_pers)
            @test d_one  isa Float64 && isfinite(d_one)
            # SVector indexed component, one-shot path
            d_sv = ForwardDiff.derivative(δ -> fn(x_f .+ δ, y_sv, xq_f)[2], 0.0)
            @test d_sv isa Float64 && isfinite(d_sv)
        end
    end

    @testset "Nested Dual — 2nd-order AD (d²/dt²)" begin
        for fn in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            d2 = ForwardDiff.derivative(
                t1 -> ForwardDiff.derivative(t2 -> fn(x_f, y_f)(t2), t1),
                xq_f,
            )
            @test d2 isa Float64
        end
    end

    @testset "ND ForwardDiff.gradient + Hessian + grid-offset sensitivity" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data = [sin(i) * cos(j) for i in 1:5, j in 1:5]
        data_sv = [SA[Float64(i + j), 2.0(i + j), 3.0(i + j)] for i in 1:5, j in 1:5]

        for fn in (linear_interp, cubic_interp)
            # Gradient of scalar-output ND interp w.r.t. query
            @test ForwardDiff.gradient(q -> fn((xg, yg), data)((q[1], q[2])), [2.5, 3.5]) isa Vector{Float64}
            # Hessian — exercises nested Dual through the kernel
            @test ForwardDiff.hessian(q -> fn((xg, yg), data)((q[1], q[2])), [2.5, 3.5]) isa Matrix{Float64}
            # Gradient via one-shot 3-arg
            @test ForwardDiff.gradient(q -> fn((xg, yg), data, (q[1], q[2])), [2.5, 3.5]) isa Vector{Float64}
            # SVector component gradient (persistent + one-shot)
            @test ForwardDiff.gradient(q -> fn((xg, yg), data_sv)((q[1], q[2]))[1], [2.5, 3.5]) isa Vector{Float64}
            @test ForwardDiff.gradient(q -> fn((xg, yg), data_sv, (q[1], q[2]))[1], [2.5, 3.5]) isa Vector{Float64}
            # ND grid-offset sensitivity via one-shot
            @test ForwardDiff.derivative(δ -> fn((xg .+ δ, yg), data, (2.5, 3.5)), 0.0) isa Float64
        end
    end

    @testset "Series + one-shot ForwardDiff.derivative on SVector component" begin
        using FastInterpolations: Series
        y_sv1 = [SA[Float64(i), 2.0i] for i in 1:10]
        y_sv2 = [SA[-1.0*i, 0.5i] for i in 1:10]
        s_sv = Series(y_sv1, y_sv2)
        # `fn(x, series, t)` returns Vector{SVector} (one per series).
        # ∂/∂t of series[1][component[1]] exercises the full Series + duck-Tq chain.
        for fn in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            d = ForwardDiff.derivative(t -> fn(x_f, s_sv, t)[1][1], xq_f)
            @test d isa Float64
        end
    end
end

# `CubicSeriesInterpolant` build-then-call path — distinct from the 3-arg
# one-shot covered in the "Series path" testitem above.
@testitem "CubicSeriesInterpolant persistent path — SVector × Dual carrier" begin
    using StaticArrays, ForwardDiff
    using FastInterpolations: Series

    x = collect(1.0:10.0)
    y_sv1 = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    y_sv2 = [SA[-1.0 * i, 0.5i, 2.5i] for i in 1:10]
    s_sv  = Series(y_sv1, y_sv2)
    xq_d  = ForwardDiff.Dual{Nothing}(5.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    sitp = cubic_interp(x, s_sv)

    @testset "scalar Dual via persistent callable" begin
        @test sitp(xq_d) isa Vector{SVector{3, D}}
    end

    @testset "batch Vector{Dual} via persistent callable" begin
        @test sitp(xq_dv) isa Vector{Vector{SVector{3, D}}}
    end
end

@testitem "Constant ND deriv-zero short-circuit propagates Tq carrier" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}

    xg = collect(1.0:5.0); yg = collect(1.0:5.0)
    data_int = [10i + j for i in 1:5, j in 1:5]
    qd = (ForwardDiff.Dual{Nothing}(2.5, 1.0), ForwardDiff.Dual{Nothing}(3.5, 0.0))

    @testset "Persistent ND scalar callable, deriv != EvalValue" begin
        itp = constant_interp((xg, yg), data_int)
        @test itp(qd; deriv = (DerivOp(1), EvalValue())) isa D
    end

    @testset "One-shot ND scalar callable, deriv != EvalValue" begin
        @test constant_interp((xg, yg), data_int, qd; deriv = (DerivOp(1), EvalValue())) isa D
    end
end

@testitem "Hermite one-shot preserves duck carrier in `dy`" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "Float64 y + Vector{Dual} dy — batch one-shot" begin
        x = collect(1.0:10.0)
        y = sin.(x)
        dy = [ForwardDiff.Dual{Nothing}(cos(xi), 1.0) for xi in x]
        xq = collect(2.0:0.5:8.0)
        @test hermite_interp(x, y, dy, xq) isa Vector{D}
    end

    @testset "ForwardDiff.derivative on slope perturbation" begin
        x = collect(1.0:10.0)
        y = sin.(x)
        dy_base = cos.(x)
        xq = 5.5
        d = ForwardDiff.derivative(δ -> hermite_interp(x, y, dy_base .+ δ, [xq])[1], 0.0)
        @test d isa Float64
        @test isfinite(d)
    end

    @testset "Float32 y + Float64 dy precision pin" begin
        # Scalar/vector parity: wider `eltype(dy)` widens the result.
        x = Float32[0, 1, 2]
        y = Float32[0, 1, 4]
        dy = Float64[0, 2, 4]
        xq_v = Float32[0.5]
        @test eltype(hermite_interp(x, y, dy, xq_v)) === Float64
    end
end

# ND deriv-zero short-circuits multiply by `one(eltype(query[1]))` (axis-1).
# Heterogeneous-axis queries like `(Float, Dual)` lose the non-axis-1 carrier
# unless the short-circuit folds `one` over every axis (mirroring the forward
# kernel's per-axis `* one(dL_d)`).
@testitem "ND deriv-zero short-circuit per-axis carrier" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}

    xg = collect(1.0:5.0); yg = collect(1.0:5.0)
    data_int = [10i + j for i in 1:5, j in 1:5]
    # Axis 1 Float, axis 2 Dual — Dual carrier lives on axis 2 only.
    q_het = (2.5, ForwardDiff.Dual{Nothing}(3.5, 1.0))
    q_het_batch = [q_het]

    @testset "Constant ND persistent scalar — (Float, Dual) × mixed deriv" begin
        itp = constant_interp((xg, yg), data_int)
        @test itp(q_het; deriv = (EvalValue(), DerivOp(1))) isa D
    end

    @testset "Constant ND one-shot scalar — (Float, Dual) × mixed deriv" begin
        @test constant_interp((xg, yg), data_int, q_het; deriv = (EvalValue(), DerivOp(1))) isa D
    end

    @testset "Constant ND one-shot batch — (Float, Dual) × mixed deriv" begin
        @test constant_interp((xg, yg), data_int, q_het_batch; deriv = (EvalValue(), DerivOp(1))) isa Vector{D}
    end

    @testset "Linear ND persistent scalar 2nd-deriv (zero-fill) — (Float, Dual)" begin
        # LinearInterpolantND _deriv_zero_fill triggers on 2nd-order deriv.
        data_f = [Float64(10i + j) for i in 1:5, j in 1:5]
        itp_l = linear_interp((xg, yg), data_f)
        @test itp_l(q_het; deriv = (EvalValue(), DerivOp(2))) isa D
    end
end

# Anchored-vector batch callable lets advanced users pre-build queries and
# reuse them across interpolants. Linear's path uses `_output_eltype(Tv, Tg, Tq)`
# for the output buffer; Constant + Quadratic only used `Vector{Tg}` so a
# duck-Tq anchor lost its carrier on `setindex!` convert.
@testitem "Anchored-vector callable propagates Tq carrier" begin
    using ForwardDiff
    using FastInterpolations: _ConstantAnchoredQuery, _QuadraticAnchoredQuery, _fill_anchors!

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    x = collect(0.0:0.1:1.0)
    y = sin.(x)
    xq_d = [ForwardDiff.Dual{Nothing}(0.15 + 0.1i, 1.0) for i in 0:4]

    @testset "Constant" begin
        itp = constant_interp(x, y)
        aq_vec = Vector{_ConstantAnchoredQuery{Float64, D}}(undef, length(xq_d))
        _fill_anchors!(aq_vec, x, xq_d, Val(:constant))
        @test itp(aq_vec) isa Vector{D}
    end

    @testset "Quadratic" begin
        itp = quadratic_interp(x, y)
        aq_vec = Vector{_QuadraticAnchoredQuery{Float64, D}}(undef, length(xq_d))
        _fill_anchors!(aq_vec, x, xq_d, Val(:quadratic))
        @test itp(aq_vec) isa Vector{D}
    end
end
