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
#   Cat C  — NaN propagation through ND in-place + persist-alloc deriv-zero
#   Cat D  — cross-path equivalence (same input → same `Dual` answer on every path)
#
# Active GREEN target (this phase): Cat C — ND Constant in-place/persist-alloc
# NaN-partials loss (codex finding + sibling persist paths). All other
# inconsistencies are pinned with `@test_broken` so they stay tracked and
# convert to `@test` one phase at a time.

@testitem "Cat A: @inferred type stability across deriv-aware paths (BROKEN pin)" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    x1 = collect(1.0:5.0); y1 = [Float64(10i) for i in 1:5]
    xq_d = ForwardDiff.Dual{Nothing}(2.5, 1.0); xq_d_b = [xq_d]

    xg = collect(1.0:5.0); yg = collect(1.0:5.0)
    d2 = [Float64(10i + j) for i in 1:5, j in 1:5]
    q_het = (2.5, ForwardDiff.Dual{Nothing}(3.5, 1.0)); q_het_b = [q_het]

    @testset "1D oneshot scalar — deriv-zero × Dual query" begin
        @test (@inferred linear_interp(x1, y1, xq_d; deriv=DerivOp(2))) isa D
        @test_broken (@inferred cubic_interp(x1, y1, xq_d; deriv=DerivOp(4))) isa D
        @test_broken (@inferred pchip_interp(x1, y1, xq_d; deriv=DerivOp(4))) isa D
        @test_broken (@inferred cardinal_interp(x1, y1, xq_d; deriv=DerivOp(4))) isa D
        @test_broken (@inferred akima_interp(x1, y1, xq_d; deriv=DerivOp(4))) isa D
    end

    # 1D vector batch paths (alloc + persist alloc) and the entire 2D matrix
    # under deriv-zero × Dual query already infer cleanly. Pin them as `@test`
    # to lock the GREEN behavior in (regression guard); the still-broken cells
    # remain on `@test_broken` above.
    @testset "1D oneshot allocating batch — deriv-zero × Dual query" begin
        @test (@inferred linear_interp(x1, y1, xq_d_b; deriv=DerivOp(2))) isa Vector{D}
        @test (@inferred cubic_interp(x1, y1, xq_d_b; deriv=DerivOp(4))) isa Vector{D}
        @test (@inferred pchip_interp(x1, y1, xq_d_b; deriv=DerivOp(4))) isa Vector{D}
    end

    @testset "1D persistent scalar — deriv-zero × Dual query" begin
        @test (@inferred linear_interp(x1, y1)(xq_d; deriv=DerivOp(2))) isa D
        @test_broken (@inferred cubic_interp(x1, y1)(xq_d; deriv=DerivOp(4))) isa D
    end

    @testset "1D persistent allocating batch — deriv-zero × Dual query" begin
        @test (@inferred linear_interp(x1, y1)(xq_d_b; deriv=DerivOp(2))) isa Vector{D}
        @test (@inferred cubic_interp(x1, y1)(xq_d_b; deriv=DerivOp(4))) isa Vector{D}
    end

    @testset "2D oneshot scalar — deriv-zero × heterogeneous Dual query" begin
        @test (@inferred linear_interp((xg, yg), d2, q_het; deriv=(EvalValue(), DerivOp(2)))) isa D
        @test (@inferred cubic_interp((xg, yg), d2, q_het; deriv=(EvalValue(), DerivOp(4)))) isa D
    end

    @testset "2D oneshot allocating batch — deriv-zero × heterogeneous Dual query" begin
        @test (@inferred constant_interp((xg, yg), d2, q_het_b; deriv=(EvalValue(), DerivOp(1)))) isa Vector{D}
        @test (@inferred linear_interp((xg, yg), d2, q_het_b; deriv=(EvalValue(), DerivOp(2)))) isa Vector{D}
        @test (@inferred cubic_interp((xg, yg), d2, q_het_b; deriv=(EvalValue(), DerivOp(4)))) isa Vector{D}
    end

    @testset "2D persistent paths — deriv-zero × heterogeneous Dual query" begin
        itp_c = constant_interp((xg, yg), d2)
        itp_l = linear_interp((xg, yg), d2)
        @test (@inferred itp_c(q_het_b; deriv=(EvalValue(), DerivOp(1)))) isa Vector{D}
        @test (@inferred itp_l(q_het_b; deriv=(EvalValue(), DerivOp(2)))) isa Vector{D}
    end
end

@testitem "Cat B: Dual carrier preserved through sub-zero deriv paths (BROKEN pin)" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    x1 = collect(1.0:5.0); y1 = [Float64(10i) for i in 1:5]
    xq_d = ForwardDiff.Dual{Nothing}(2.5, 1.0)

    # 1D scalar paths drop Dual → Float64 for non-zero deriv ≥ 1 when the
    # kernel branch doesn't multiply through the query carrier. Linear's
    # kernels now carry `* one(α)` (Phase 2) — Cubic/Quadratic/PCHIP still
    # pending.
    @testset "1D oneshot scalar Dual return for non-zero deriv" begin
        @test linear_interp(x1, y1, xq_d; deriv=DerivOp(1)) isa D
        @test_broken quadratic_interp(x1, y1, xq_d; deriv=DerivOp(2)) isa D
        @test_broken cubic_interp(x1, y1, xq_d; deriv=DerivOp(3)) isa D
        @test_broken pchip_interp(x1, y1, xq_d; deriv=DerivOp(3)) isa D
    end

    @testset "1D persistent scalar Dual return for non-zero deriv" begin
        @test linear_interp(x1, y1)(xq_d; deriv=DerivOp(1)) isa D
        @test_broken cubic_interp(x1, y1)(xq_d; deriv=DerivOp(3)) isa D
    end

    @testset "1D oneshot scalar Dual return for zero-order deriv" begin
        @test linear_interp(x1, y1, xq_d; deriv=DerivOp(2)) isa D
        @test_broken cubic_interp(x1, y1, xq_d; deriv=DerivOp(4)) isa D
        @test_broken pchip_interp(x1, y1, xq_d; deriv=DerivOp(4)) isa D
    end
end

@testitem "Cat C: ND deriv-zero NaN propagation through Dual carrier" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    xg = collect(1.0:5.0); yg = collect(1.0:5.0)
    # NaN at first(data) drives `0 * first(data) = NaN`; if the deriv-zero
    # short-circuit forwards the carrier (sample-shaped Dual), the NaN must
    # propagate into both `value` AND `partials`.
    data_nan = [Float64(10i + j) for i in 1:5, j in 1:5]; data_nan[1, 1] = NaN

    q_het = (2.5, ForwardDiff.Dual{Nothing}(3.5, 1.0)); q_het_b = [q_het]
    dv = (EvalValue(), DerivOp(1))   # Constant: any deriv ≥ 1 is zero-fill

    # ── Reference rows: scalar paths already propagate NaN end-to-end ──
    @testset "ND Constant oneshot scalar — reference (already GREEN)" begin
        r = constant_interp((xg, yg), data_nan, q_het; deriv=dv)
        @test r isa D
        @test isnan(ForwardDiff.value(r))
        @test isnan(ForwardDiff.partials(r)[1])
    end

    @testset "ND Constant oneshot allocating — reference (already GREEN)" begin
        r = constant_interp((xg, yg), data_nan, q_het_b; deriv=dv)
        @test r isa Vector{D}
        @test isnan(ForwardDiff.value(r[1]))
        @test isnan(ForwardDiff.partials(r[1])[1])
    end

    @testset "ND Constant persist scalar — reference (already GREEN)" begin
        itp = constant_interp((xg, yg), data_nan)
        r = itp(q_het; deriv=dv)
        @test r isa D
        @test isnan(ForwardDiff.value(r))
        @test isnan(ForwardDiff.partials(r)[1])
    end

    # ── ACTIVE RED rows: the codex finding + sibling persist paths ──
    @testset "ND Constant oneshot in-place — NaN partials must propagate" begin
        o = Vector{D}(undef, 1)
        constant_interp!(o, (xg, yg), data_nan, q_het_b; deriv=dv)
        @test isnan(ForwardDiff.value(o[1]))
        @test isnan(ForwardDiff.partials(o[1])[1])    # RED: currently 0.0
    end

    @testset "ND Constant persist allocating — NaN must propagate" begin
        itp = constant_interp((xg, yg), data_nan)
        r = itp(q_het_b; deriv=dv)
        @test r isa Vector{D}
        @test isnan(ForwardDiff.value(r[1]))           # RED: currently 0.0
        @test isnan(ForwardDiff.partials(r[1])[1])
    end

    @testset "ND Constant persist in-place — NaN partials must propagate" begin
        itp = constant_interp((xg, yg), data_nan)
        o = Vector{D}(undef, 1)
        itp(o, q_het_b; deriv=dv)
        @test isnan(ForwardDiff.value(o[1]))           # RED: currently 0.0
        @test isnan(ForwardDiff.partials(o[1])[1])
    end

    # ── BROKEN pin: same invariant for other ND methods (next phase) ──
    @testset "ND Linear deriv=2 in-place NaN partials (broken pin)" begin
        data_f = [Float64(10i + j) for i in 1:5, j in 1:5]; data_f[1, 1] = NaN
        dv_lin = (EvalValue(), DerivOp(2))
        o = Vector{D}(undef, 1)
        linear_interp!(o, (xg, yg), data_f, q_het_b; deriv=dv_lin)
        @test_broken isnan(ForwardDiff.value(o[1])) && isnan(ForwardDiff.partials(o[1])[1])
    end
end

@testitem "Cat D: Cross-path equivalence under identical inputs (BROKEN pin)" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    xg = collect(1.0:5.0); yg = collect(1.0:5.0)
    data_nan = [Float64(10i + j) for i in 1:5, j in 1:5]; data_nan[1, 1] = NaN
    q_het = (2.5, ForwardDiff.Dual{Nothing}(3.5, 1.0)); q_het_b = [q_het]
    dv = (EvalValue(), DerivOp(1))

    # Compute each path independently, then assert pairwise equivalence.
    ref_scalar    = constant_interp((xg, yg), data_nan, q_het; deriv=dv)
    alloc_first   = constant_interp((xg, yg), data_nan, q_het_b; deriv=dv)[1]

    inplace_buf   = Vector{D}(undef, 1)
    constant_interp!(inplace_buf, (xg, yg), data_nan, q_het_b; deriv=dv)
    inplace_first = inplace_buf[1]

    itp = constant_interp((xg, yg), data_nan)
    persist_scalar  = itp(q_het; deriv=dv)
    persist_alloc   = itp(q_het_b; deriv=dv)[1]
    persist_in_buf  = Vector{D}(undef, 1); itp(persist_in_buf, q_het_b; deriv=dv)
    persist_inplace = persist_in_buf[1]

    @testset "scalar ↔ allocating batch (already GREEN)" begin
        @test isequal(ref_scalar, alloc_first)
    end

    @testset "scalar ↔ in-place batch — RED (codex)" begin
        @test isequal(ref_scalar, inplace_first)
    end

    @testset "scalar ↔ persistent scalar — already GREEN" begin
        @test isequal(ref_scalar, persist_scalar)
    end

    @testset "scalar ↔ persistent allocating — RED" begin
        @test isequal(ref_scalar, persist_alloc)
    end

    @testset "scalar ↔ persistent in-place — RED" begin
        @test isequal(ref_scalar, persist_inplace)
    end
end
