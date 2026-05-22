# Carrier propagation + `@inferred` type-stability invariants across the
# full path matrix: 7 methods × {1D, 2D} × {oneshot, persistent} × {scalar,
# allocating batch, in-place batch}.
#
# Two assertions are folded into one expression — `@test (@inferred f()) isa T`
# — so each test enforces BOTH return-type inference (compiler proof that the
# call is type-stable) AND result-shape (Dual carrier preserved through the
# pipeline). Either fault alone breaks the test.
#
# Categories
# ----------
#   Cat A  — `@inferred` type stability across deriv-aware paths
#   Cat B  — Dual query → Dual result (carrier preservation, sub-zero deriv)
#   Cat C  — Cell-local NaN propagation through ND deriv-zero short-circuits
#   Cat D  — cross-path equivalence (same input → same `Dual` answer on every path)

@testitem "Cat A: @inferred type stability across deriv-aware paths" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    x1 = collect(1.0:5.0)
    y1 = [Float64(10i) for i in 1:5]
    xq_d = ForwardDiff.Dual{Nothing}(2.5, 1.0)
    xq_d_b = [xq_d]

    xg = collect(1.0:5.0)
    yg = collect(1.0:5.0)
    d2 = [Float64(10i + j) for i in 1:5, j in 1:5]
    q_het = (2.5, ForwardDiff.Dual{Nothing}(3.5, 1.0))
    q_het_b = [q_het]

    @testset "1D oneshot scalar — deriv-zero × Dual query" begin
        @test (@inferred linear_interp(x1, y1, xq_d; deriv = DerivOp(2))) isa D
        @test (@inferred cubic_interp(x1, y1, xq_d; deriv = DerivOp(4))) isa D
        @test (@inferred pchip_interp(x1, y1, xq_d; deriv = DerivOp(4))) isa D
        @test (@inferred cardinal_interp(x1, y1, xq_d; deriv = DerivOp(4))) isa D
        @test (@inferred akima_interp(x1, y1, xq_d; deriv = DerivOp(4))) isa D
    end

    @testset "1D oneshot allocating batch — deriv-zero × Dual query" begin
        @test (@inferred linear_interp(x1, y1, xq_d_b; deriv = DerivOp(2))) isa Vector{D}
        @test (@inferred cubic_interp(x1, y1, xq_d_b; deriv = DerivOp(4))) isa Vector{D}
        @test (@inferred pchip_interp(x1, y1, xq_d_b; deriv = DerivOp(4))) isa Vector{D}
    end

    @testset "1D persistent scalar — deriv-zero × Dual query" begin
        @test (@inferred linear_interp(x1, y1)(xq_d; deriv = DerivOp(2))) isa D
        @test (@inferred cubic_interp(x1, y1)(xq_d; deriv = DerivOp(4))) isa D
    end

    @testset "1D persistent allocating batch — deriv-zero × Dual query" begin
        @test (@inferred linear_interp(x1, y1)(xq_d_b; deriv = DerivOp(2))) isa Vector{D}
        @test (@inferred cubic_interp(x1, y1)(xq_d_b; deriv = DerivOp(4))) isa Vector{D}
    end

    @testset "2D oneshot scalar — deriv-zero × heterogeneous Dual query" begin
        @test (@inferred linear_interp((xg, yg), d2, q_het; deriv = (EvalValue(), DerivOp(2)))) isa D
        @test (@inferred cubic_interp((xg, yg), d2, q_het; deriv = (EvalValue(), DerivOp(4)))) isa D
    end

    @testset "2D oneshot allocating batch — deriv-zero × heterogeneous Dual query" begin
        @test (@inferred constant_interp((xg, yg), d2, q_het_b; deriv = (EvalValue(), DerivOp(1)))) isa Vector{D}
        @test (@inferred linear_interp((xg, yg), d2, q_het_b; deriv = (EvalValue(), DerivOp(2)))) isa Vector{D}
        @test (@inferred cubic_interp((xg, yg), d2, q_het_b; deriv = (EvalValue(), DerivOp(4)))) isa Vector{D}
    end

    @testset "2D persistent paths — deriv-zero × heterogeneous Dual query" begin
        itp_c = constant_interp((xg, yg), d2)
        itp_l = linear_interp((xg, yg), d2)
        @test (@inferred itp_c(q_het_b; deriv = (EvalValue(), DerivOp(1)))) isa Vector{D}
        @test (@inferred itp_l(q_het_b; deriv = (EvalValue(), DerivOp(2)))) isa Vector{D}
    end
end

@testitem "Cat B: Dual carrier preserved through sub-zero deriv paths" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    x1 = collect(1.0:5.0)
    y1 = [Float64(10i) for i in 1:5]
    xq_d = ForwardDiff.Dual{Nothing}(2.5, 1.0)

    # 1D scalar paths must keep the Dual carrier for non-zero deriv ≥ 1
    # and for zero-order (deriv beyond the polynomial degree). The kernel
    # branches carry `* one(α)` (or `* one(dL)`) so the query carrier
    # threads through even when the math is constant w.r.t. the query.
    @testset "1D oneshot scalar Dual return for non-zero deriv" begin
        @test linear_interp(x1, y1, xq_d; deriv = DerivOp(1)) isa D
        @test quadratic_interp(x1, y1, xq_d; deriv = DerivOp(2)) isa D
        @test cubic_interp(x1, y1, xq_d; deriv = DerivOp(3)) isa D
        @test pchip_interp(x1, y1, xq_d; deriv = DerivOp(3)) isa D
    end

    @testset "1D persistent scalar Dual return for non-zero deriv" begin
        @test linear_interp(x1, y1)(xq_d; deriv = DerivOp(1)) isa D
        @test cubic_interp(x1, y1)(xq_d; deriv = DerivOp(3)) isa D
    end

    @testset "1D oneshot scalar Dual return for zero-order deriv" begin
        @test linear_interp(x1, y1, xq_d; deriv = DerivOp(2)) isa D
        @test cubic_interp(x1, y1, xq_d; deriv = DerivOp(4)) isa D
        @test pchip_interp(x1, y1, xq_d; deriv = DerivOp(4)) isa D
    end
end

# NaN-propagation contract is *cell-local* — only NaN in the data cells
# touched by the query may reach the result. NaN elsewhere in the array
# stays invisible. This holds even for deriv-zero short-circuits: the
# kernel result is multiplied by `0` at the cell-local stage, so IEEE
# `NaN * 0 = NaN` carries the NaN through value + partials slots.
#
# Scope: cell-local only applies to *kernel-local* methods (Constant, Linear,
# Hermite-family). Cubic / Quadratic perform a *global tridiagonal solve*
# at build time, so a single NaN in `data` oxidizes *every* coefficient —
# their NaN behavior is "global" rather than cell-local, and they're not
# covered by this contract.
@testitem "Cat C: Cell-local NaN propagation through ND deriv-zero" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    xg = collect(1.0:5.0)
    yg = collect(1.0:5.0)

    # Query (2.5, 3.5) lies in the cell whose 2D corner indices are
    # {(2,3), (3,3), (2,4), (3,4)}.
    q_het = (2.5, ForwardDiff.Dual{Nothing}(3.5, 1.0))
    q_het_b = [q_het]

    # Cell-local NaN: at a corner the query's cell actually reads.
    cell_data = [Float64(10i + j) for i in 1:5, j in 1:5]
    cell_data[2, 3] = NaN

    # Out-of-cell NaN: far from the query's cell. Must NOT reach the result.
    far_data = [Float64(10i + j) for i in 1:5, j in 1:5]
    far_data[1, 1] = NaN

    dv = (EvalValue(), DerivOp(1))   # Constant: any deriv ≥ 1 is zero-fill

    # ── Cell-local NaN MUST propagate (value + partials both NaN) ──
    @testset "Constant ND oneshot scalar — cell-local NaN propagates" begin
        r = constant_interp((xg, yg), cell_data, q_het; deriv = dv)
        @test r isa D
        @test isnan(ForwardDiff.value(r))
        @test isnan(ForwardDiff.partials(r)[1])
    end

    @testset "Constant ND oneshot allocating — cell-local NaN propagates" begin
        r = constant_interp((xg, yg), cell_data, q_het_b; deriv = dv)
        @test r isa Vector{D}
        @test isnan(ForwardDiff.value(r[1]))
        @test isnan(ForwardDiff.partials(r[1])[1])
    end

    @testset "Constant ND oneshot in-place — cell-local NaN propagates" begin
        o = Vector{D}(undef, 1)
        constant_interp!(o, (xg, yg), cell_data, q_het_b; deriv = dv)
        @test isnan(ForwardDiff.value(o[1]))
        @test isnan(ForwardDiff.partials(o[1])[1])
    end

    @testset "Constant ND persist scalar — cell-local NaN propagates" begin
        itp = constant_interp((xg, yg), cell_data)
        r = itp(q_het; deriv = dv)
        @test r isa D
        @test isnan(ForwardDiff.value(r))
        @test isnan(ForwardDiff.partials(r)[1])
    end

    @testset "Constant ND persist allocating — cell-local NaN propagates" begin
        itp = constant_interp((xg, yg), cell_data)
        r = itp(q_het_b; deriv = dv)
        @test r isa Vector{D}
        @test isnan(ForwardDiff.value(r[1]))
        @test isnan(ForwardDiff.partials(r[1])[1])
    end

    @testset "Constant ND persist in-place — cell-local NaN propagates" begin
        itp = constant_interp((xg, yg), cell_data)
        o = Vector{D}(undef, 1)
        itp(o, q_het_b; deriv = dv)
        @test isnan(ForwardDiff.value(o[1]))
        @test isnan(ForwardDiff.partials(o[1])[1])
    end

    # ── Out-of-cell NaN MUST NOT propagate (result is a clean zero) ──
    @testset "Constant ND oneshot scalar — out-of-cell NaN stays hidden" begin
        r = constant_interp((xg, yg), far_data, q_het; deriv = dv)
        @test r isa D
        @test ForwardDiff.value(r) == 0.0
        @test ForwardDiff.partials(r)[1] == 0.0
    end

    @testset "Constant ND oneshot in-place — out-of-cell NaN stays hidden" begin
        o = Vector{D}(undef, 1)
        constant_interp!(o, (xg, yg), far_data, q_het_b; deriv = dv)
        @test ForwardDiff.value(o[1]) == 0.0
        @test ForwardDiff.partials(o[1])[1] == 0.0
    end

    @testset "Constant ND persist in-place — out-of-cell NaN stays hidden" begin
        itp = constant_interp((xg, yg), far_data)
        o = Vector{D}(undef, 1)
        itp(o, q_het_b; deriv = dv)
        @test ForwardDiff.value(o[1]) == 0.0
        @test ForwardDiff.partials(o[1])[1] == 0.0
    end

    # ── Linear ND deriv-zero (deriv ≥ 2) inherits the same contract ──
    # Linear's `_linear_weight(::EvalDeriv2+, ...) = zero(α)` produces a
    # carrier-aware zero per corner; the multilinear corner sum threads the
    # cell's NaN through IEEE multiplication.
    @testset "Linear ND oneshot in-place — cell-local NaN propagates" begin
        dv_lin = (EvalValue(), DerivOp(2))
        o = Vector{D}(undef, 1)
        linear_interp!(o, (xg, yg), cell_data, q_het_b; deriv = dv_lin)
        @test isnan(ForwardDiff.value(o[1]))
        @test isnan(ForwardDiff.partials(o[1])[1])
    end
end

@testitem "Cat D: Cross-path equivalence under cell-local NaN" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    xg = collect(1.0:5.0)
    yg = collect(1.0:5.0)
    cell_data = [Float64(10i + j) for i in 1:5, j in 1:5]
    cell_data[2, 3] = NaN  # corner of query (2.5, 3.5)'s cell

    q_het = (2.5, ForwardDiff.Dual{Nothing}(3.5, 1.0))
    q_het_b = [q_het]
    dv = (EvalValue(), DerivOp(1))

    # Compute each path independently, then assert pairwise equivalence —
    # every path must surface the cell-local NaN identically.
    ref_scalar = constant_interp((xg, yg), cell_data, q_het; deriv = dv)
    alloc_first = constant_interp((xg, yg), cell_data, q_het_b; deriv = dv)[1]

    inplace_buf = Vector{D}(undef, 1)
    constant_interp!(inplace_buf, (xg, yg), cell_data, q_het_b; deriv = dv)
    inplace_first = inplace_buf[1]

    itp = constant_interp((xg, yg), cell_data)
    persist_scalar = itp(q_het; deriv = dv)
    persist_alloc = itp(q_het_b; deriv = dv)[1]
    persist_in_buf = Vector{D}(undef, 1)
    itp(persist_in_buf, q_het_b; deriv = dv)
    persist_inplace = persist_in_buf[1]

    @testset "scalar ↔ allocating batch" begin
        @test isequal(ref_scalar, alloc_first)
    end

    @testset "scalar ↔ in-place batch" begin
        @test isequal(ref_scalar, inplace_first)
    end

    @testset "scalar ↔ persistent scalar" begin
        @test isequal(ref_scalar, persist_scalar)
    end

    @testset "scalar ↔ persistent allocating" begin
        @test isequal(ref_scalar, persist_alloc)
    end

    @testset "scalar ↔ persistent in-place" begin
        @test isequal(ref_scalar, persist_inplace)
    end
end
