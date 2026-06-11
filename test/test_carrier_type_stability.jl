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
#   Cat E  — OOB cell-local NaN through `_promote_extrap_zero` (Number case)
#   Cat F  — Constant 1D right-edge cell-local NaN (Series subtests pinned `@test_broken`)
#   Cat G  — Hetero `NoInterp` deriv cell-local NaN (`zero(Tz)` → `data[slice] * zero(Tz)`)

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
        @test (@inferred quadratic_interp(x1, y1, xq_d; deriv = DerivOp(3))) isa D
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
        @test (@inferred quadratic_interp(x1, y1)(xq_d; deriv = DerivOp(3))) isa D
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

    # ND counterpart of "non-zero deriv". Cat A covers zero-fill paths
    # (Linear D2+, Cubic D4+ — `_linear_weight(::EvalDeriv2+) = zero(α)`
    # carries Tq via the zero itself). Below pin the *kernel* branch that
    # mathematically returns a non-zero gradient: Linear D1's per-corner
    # `±inv_h` weight, Cubic D3, Quadratic D2 etc. — these must still
    # thread Tq even though `α` does not appear in the weight expression.
    #
    # Helpers below take `fn::F` so `@inferred` specializes per concrete
    # method; iterating without them collapses `fn` to a Union and breaks
    # the inference check.
    xg = collect(1.0:5.0)
    yg = collect(1.0:5.0)
    d2 = [Float64(10i + j) for i in 1:5, j in 1:5]
    # (Float, Dual) — Tq lives on axis 2 only (per-axis carrier).
    q_het = (2.5, ForwardDiff.Dual{Nothing}(3.5, 1.0))
    # (Dual, Dual) — Tq on both axes (mixed-partial weight product).
    q_both = (ForwardDiff.Dual{Nothing}(2.5, 1.0), ForwardDiff.Dual{Nothing}(3.5, 1.0))
    q_het_b = [q_het]
    q_both_b = [q_both]

    # (function, axis-2 non-zero deriv level for the `(EvalValue, Dk)` pattern)
    nd_methods_nonzero = (
        (linear_interp, DerivOp(1)),
        (cubic_interp, DerivOp(3)),
        (quadratic_interp, DerivOp(2)),
        (pchip_interp, DerivOp(3)),
        (cardinal_interp, DerivOp(3)),
        (akima_interp, DerivOp(3)),
    )

    function _check_oneshot_scalar(::Type{Dt}, fn::F, grids, data, qh, qb, dk) where {F, Dt}
        @test (@inferred fn(grids, data, qh; deriv = (EvalValue(), dk))) isa Dt
        @test (@inferred fn(grids, data, qb; deriv = (DerivOp(1), DerivOp(1)))) isa Dt
    end
    function _check_oneshot_batch(::Type{Dt}, fn::F, grids, data, qhb, qbb, dk) where {F, Dt}
        @test (@inferred fn(grids, data, qhb; deriv = (EvalValue(), dk))) isa Vector{Dt}
        @test (@inferred fn(grids, data, qbb; deriv = (DerivOp(1), DerivOp(1)))) isa Vector{Dt}
    end
    function _check_persistent(::Type{Dt}, fn::F, grids, data, qh, qb, qhb, qbb, dk) where {F, Dt}
        itp = fn(grids, data)
        @test (@inferred itp(qh; deriv = (EvalValue(), dk))) isa Dt
        @test (@inferred itp(qb; deriv = (DerivOp(1), DerivOp(1)))) isa Dt
        @test (@inferred itp(qhb; deriv = (EvalValue(), dk))) isa Vector{Dt}
        @test (@inferred itp(qbb; deriv = (DerivOp(1), DerivOp(1)))) isa Vector{Dt}
    end

    @testset "ND oneshot scalar — non-zero deriv (per-axis + mixed-partial)" begin
        for (fn, dk) in nd_methods_nonzero
            _check_oneshot_scalar(D, fn, (xg, yg), d2, q_het, q_both, dk)
        end
    end

    # The Linear `_linear_weight(::EvalDeriv1)` bug surfaced here first as
    # `TypeError` — batch buffer is allocated as `Vector{Dt}`, the kernel
    # returned plain `Tv`, so the per-query store failed the typeassert.
    @testset "ND oneshot batch — non-zero deriv (Linear bug surface site)" begin
        for (fn, dk) in nd_methods_nonzero
            _check_oneshot_batch(D, fn, (xg, yg), d2, q_het_b, q_both_b, dk)
        end
    end

    @testset "ND persistent — non-zero deriv (scalar + batch)" begin
        for (fn, dk) in nd_methods_nonzero
            _check_persistent(D, fn, (xg, yg), d2, q_het, q_both, q_het_b, q_both_b, dk)
        end
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

    # Linear ND: kernel-native cell-local via `_linear_weight(::EvalDeriv2+) = zero(α)`.
    @testset "Linear ND — scalar ↔ in-place batch" begin
        ref_l = linear_interp((xg, yg), cell_data, q_het; deriv = (EvalValue(), DerivOp(2)))
        buf_l = Vector{D}(undef, 1)
        linear_interp!(buf_l, (xg, yg), cell_data, q_het_b; deriv = (EvalValue(), DerivOp(2)))
        @test isequal(ref_l, buf_l[1])
    end
end

# OOB extrap × deriv contract (cell-local "OOB cell = fill-value data"):
#
#   ClampExtrap: OOB cell's data = `y_bnd` (boundary y value). Deriv = `0 * y_bnd`
#                — NaN at `y_bnd` propagates (boundary cell-local NaN).
#   FillExtrap:  OOB cell's data = `e.fill_value`. Deriv = `0 * fill_value`
#                — NaN `fill_value` propagates, finite `fill_value` → 0.
#                Boundary `y_bnd` is NOT consulted (fill_value IS the data
#                in the OOB region).
#
# This mirrors the strict cell-local interpretation: OOB data is whatever
# the extrap rule fills there, and derivative is computed from THAT data.
@testitem "Cat E: OOB extrap × deriv — fill_value-as-data contract" begin
    using ForwardDiff
    using FastInterpolations: _promote_extrap_zero

    methods_1d = (
        linear_interp, constant_interp, cubic_interp, quadratic_interp,
        pchip_interp, cardinal_interp, akima_interp,
    )

    @testset "_promote_extrap_zero helper — Number vs Array consistency" begin
        # NaN at the data source propagates through deriv-zero promotion.
        res_arr = @inferred _promote_extrap_zero([NaN, 1.0], 0.5)
        @test isnan(res_arr[1])
        @test (@inferred _promote_extrap_zero(NaN, 0.5)) isa Float64
        @test isnan(_promote_extrap_zero(NaN, 0.5))

        # Carrier preserved + primal NaN propagated through Dual xq.
        xq_d = ForwardDiff.Dual{Nothing}(0.5, 1.0)
        res_d = @inferred _promote_extrap_zero(NaN, xq_d)
        @test res_d isa typeof(xq_d)
        @test isnan(ForwardDiff.value(res_d))
    end

    # ── ClampExtrap (unchanged): y_bnd is the OOB data ──
    @testset "1D OOB ClampExtrap × deriv: queried-boundary y NaN propagates" begin
        x = collect(1.0:5.0)
        y_nan_left = [NaN, 20.0, 30.0, 40.0, 50.0]
        y_nan_right = [10.0, 20.0, 30.0, 40.0, NaN]
        xq_oob_lo, xq_oob_hi = -1.0, 7.0

        for method in methods_1d
            @test isnan(method(x, y_nan_left, xq_oob_lo; extrap = ClampExtrap(), deriv = DerivOp(1)))
            @test isnan(method(x, y_nan_right, xq_oob_hi; extrap = ClampExtrap(), deriv = DerivOp(1)))
            # Cell-local: NaN at the OTHER boundary does NOT propagate.
            @test !isnan(method(x, y_nan_left, xq_oob_hi; extrap = ClampExtrap(), deriv = DerivOp(1)))
            @test !isnan(method(x, y_nan_right, xq_oob_lo; extrap = ClampExtrap(), deriv = DerivOp(1)))
        end
    end

    # ── FillExtrap: fill_value is the OOB data, y_bnd is IGNORED ──
    @testset "1D OOB FillExtrap(NaN) × deriv: fill_value NaN propagates" begin
        x = collect(1.0:5.0)
        y_finite = [10.0, 20.0, 30.0, 40.0, 50.0]
        for method in methods_1d
            # fill_value = NaN → deriv = NaN (regardless of finite y_bnd)
            @test isnan(method(x, y_finite, -1.0; extrap = FillExtrap(NaN), deriv = DerivOp(1)))
            @test isnan(method(x, y_finite, 7.0; extrap = FillExtrap(NaN), deriv = DerivOp(1)))
        end
    end

    @testset "1D OOB FillExtrap(finite) × deriv: returns 0 (fill_value × 0)" begin
        x = collect(1.0:5.0)
        y_finite = [10.0, 20.0, 30.0, 40.0, 50.0]
        for method in methods_1d
            @test method(x, y_finite, -1.0; extrap = FillExtrap(99.0), deriv = DerivOp(1)) == 0.0
            @test method(x, y_finite, 7.0; extrap = FillExtrap(99.0), deriv = DerivOp(1)) == 0.0
        end
    end

    @testset "1D OOB FillExtrap × deriv: y_bnd NaN is IGNORED" begin
        # Key new contract: under FillExtrap, y_bnd is not the OOB data —
        # fill_value is. y_bnd NaN must NOT propagate through deriv.
        x = collect(1.0:5.0)
        y_nan_left = [NaN, 20.0, 30.0, 40.0, 50.0]
        y_nan_right = [10.0, 20.0, 30.0, 40.0, NaN]
        for method in methods_1d
            @test method(x, y_nan_left, -1.0; extrap = FillExtrap(0.0), deriv = DerivOp(1)) == 0.0
            @test method(x, y_nan_right, 7.0; extrap = FillExtrap(0.0), deriv = DerivOp(1)) == 0.0
        end
    end

    # ── ND OOB FillExtrap × deriv: fill_value-as-data ──
    @testset "ND OOB FillExtrap × deriv: fill_value-as-data" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data = [Float64(10i + j) for i in 1:5, j in 1:5]
        q_oob = (7.0, 3.5)

        # finite fill_value × deriv → 0 (any axis with deriv)
        @test constant_interp(
            (xg, yg), data, q_oob; extrap = FillExtrap(99.0),
            deriv = (DerivOp(1), EvalValue())
        ) == 0.0
        @test constant_interp(
            (xg, yg), data, q_oob; extrap = FillExtrap(99.0),
            deriv = (EvalValue(), DerivOp(1))
        ) == 0.0
        @test constant_interp(
            (xg, yg), data, [q_oob]; extrap = FillExtrap(99.0),
            deriv = (DerivOp(1), EvalValue())
        ) == [0.0]
        itp99 = constant_interp((xg, yg), data; extrap = FillExtrap(99.0))
        @test itp99(q_oob; deriv = (DerivOp(1), EvalValue())) == 0.0

        # NaN fill_value × deriv → NaN
        @test isnan(
            constant_interp(
                (xg, yg), data, q_oob; extrap = FillExtrap(NaN),
                deriv = (DerivOp(1), EvalValue())
            )
        )
        @test isnan(
            constant_interp(
                (xg, yg), data, q_oob; extrap = FillExtrap(NaN),
                deriv = (EvalValue(), DerivOp(1))
            )
        )
        @test isnan(
            constant_interp(
                (xg, yg), data, [q_oob]; extrap = FillExtrap(NaN),
                deriv = (DerivOp(1), EvalValue())
            )[1]
        )
        itp_nan = constant_interp((xg, yg), data; extrap = FillExtrap(NaN))
        @test isnan(itp_nan(q_oob; deriv = (DerivOp(1), EvalValue())))

        # Same contract on Linear ND (non-Constant) — sanity that the
        # `_fill_extrap_result` dispatch applies uniformly.
        @test linear_interp(
            (xg, yg), data, q_oob; extrap = FillExtrap(99.0),
            deriv = (DerivOp(1), EvalValue())
        ) == 0.0
        @test isnan(
            linear_interp(
                (xg, yg), data, q_oob; extrap = FillExtrap(NaN),
                deriv = (DerivOp(1), EvalValue())
            )
        )
    end

    # ── Hetero mixed Real + NoInterp: OOB FillExtrap × NoInterp deriv ──
    # The Real-axis OOB result IS the fill_value-as-data (no boundary y leak),
    # and the NoInterp axis's `* 0` carries IEEE multiplication.
    @testset "Hetero mixed OOB FillExtrap × NoInterp deriv: fill_value-as-data" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data = [Float64(10i + j) for i in 1:5, j in 1:5]
        q_oob = (7.0, GridIdx(2))  # Real-axis 7.0 OOB; NoInterp idx 2 in-domain

        # finite fill_value → 0 (oneshot + persistent)
        @test interp(
            (xg, yg), data, q_oob;
            method = (LinearInterp(), NoInterp()),
            extrap = FillExtrap(99.0),
            deriv = (EvalValue(), DerivOp(1))
        ) == 0.0
        itp99 = interp(
            (xg, yg), data;
            method = (LinearInterp(), NoInterp()),
            extrap = FillExtrap(99.0)
        )
        @test itp99(q_oob; deriv = (EvalValue(), DerivOp(1))) == 0.0

        # NaN fill_value → NaN (oneshot + persistent)
        @test isnan(
            interp(
                (xg, yg), data, q_oob;
                method = (LinearInterp(), NoInterp()),
                extrap = FillExtrap(NaN),
                deriv = (EvalValue(), DerivOp(1))
            )
        )
        itp_nan = interp(
            (xg, yg), data;
            method = (LinearInterp(), NoInterp()),
            extrap = FillExtrap(NaN)
        )
        @test isnan(itp_nan(q_oob; deriv = (EvalValue(), DerivOp(1))))
    end
end

# Constant 1D right-edge + Series in-domain / boundary deriv paths route
# through `_constant_kernel(op, ...)` (1D) or per-k cell-local `0 * y[idx, k]
# * one(aq.dL)` (Series), so cell-local NaN propagates through deriv-zero
# in parallel to the value path (Cat E pins the OOB extrap leg).
@testitem "Cat F: Constant 1D right-edge + Series cell-local NaN" begin
    x = collect(1.0:5.0)
    y_nan_right = [10.0, 20.0, 30.0, 40.0, NaN]
    y_nan_left = [NaN, 20.0, 30.0, 40.0, 50.0]
    xi_edge = 5.0  # == last(x), triggers the right-edge short-circuit

    @testset "1D oneshot scalar — right-edge cell-local" begin
        # NaN at right-edge boundary cell propagates through deriv-zero.
        @test isnan(constant_interp(x, y_nan_right, xi_edge; deriv = DerivOp(1)))
        # NaN at far (left) boundary does NOT leak through right-edge query.
        @test !isnan(constant_interp(x, y_nan_left, xi_edge; deriv = DerivOp(1)))
        # ClampExtrap also routes through the same short-circuit when in-domain.
        @test isnan(constant_interp(x, y_nan_right, xi_edge; extrap = ClampExtrap(), deriv = DerivOp(1)))
        @test !isnan(constant_interp(x, y_nan_left, xi_edge; extrap = ClampExtrap(), deriv = DerivOp(1)))
        # WrapExtrap right-edge short-circuit (`xi_wrapped == last(x)`)
        @test isnan(constant_interp(x, y_nan_right, xi_edge; extrap = WrapExtrap(), deriv = DerivOp(1)))
        @test !isnan(constant_interp(x, y_nan_left, xi_edge; extrap = WrapExtrap(), deriv = DerivOp(1)))
    end

    @testset "1D persistent scalar — right-edge cell-local" begin
        itp_right = constant_interp(x, y_nan_right)
        itp_left = constant_interp(x, y_nan_left)
        @test isnan(itp_right(xi_edge; deriv = DerivOp(1)))
        @test !isnan(itp_left(xi_edge; deriv = DerivOp(1)))
        # Same for ClampExtrap / FillExtrap variants of the anchor path.
        itp_right_clamp = constant_interp(x, y_nan_right; extrap = ClampExtrap())
        itp_left_clamp = constant_interp(x, y_nan_left; extrap = ClampExtrap())
        @test isnan(itp_right_clamp(xi_edge; deriv = DerivOp(1)))
        @test !isnan(itp_left_clamp(xi_edge; deriv = DerivOp(1)))
        itp_right_fill = constant_interp(x, y_nan_right; extrap = FillExtrap(0.0))
        itp_left_fill = constant_interp(x, y_nan_left; extrap = FillExtrap(0.0))
        @test isnan(itp_right_fill(xi_edge; deriv = DerivOp(1)))
        @test !isnan(itp_left_fill(xi_edge; deriv = DerivOp(1)))
    end

    @testset "1D anchored-query — right-edge cell-local (all extraps)" begin
        # Hits `_constant_anchor_dispatch` (NoExtrap variant has its own
        # right-edge branch separate from the shared `_constant_eval_at_anchor`).
        aq = FastInterpolations._anchor_query(x, xi_edge, Val(:constant))
        for extrap in (NoExtrap(), ClampExtrap(), FillExtrap(0.0))
            itp_r = constant_interp(x, y_nan_right; extrap)
            itp_l = constant_interp(x, y_nan_left; extrap)
            @test isnan(itp_r(aq; deriv = DerivOp(1)))
            @test !isnan(itp_l(aq; deriv = DerivOp(1)))
        end
    end

    @testset "Series scalar — right-boundary cell-local per series" begin
        Y = [Float64(10i + j) for i in 1:5, j in 1:3]
        Y[5, 2] = NaN  # right boundary, series 2 only
        sitp = constant_interp(x, Series(Y))
        res = sitp(xi_edge; deriv = DerivOp(1))
        @test !isnan(res[1])
        @test isnan(res[2])
        @test !isnan(res[3])
    end

    @testset "Series batch — in-domain cell-local per series" begin
        Y = [Float64(10i + j) for i in 1:5, j in 1:3]
        Y[3, 2] = NaN  # interior cell corner, series 2 only
        sitp = constant_interp(x, Series(Y))
        xq_batch = [2.5, 3.5]  # 3.5 hits cell (3, 4); 2.5 hits cell (2, 3)
        res = sitp(xq_batch; deriv = DerivOp(1))
        # Layout: res[k][j] — outer per series, inner per query.
        @test !isnan(res[1][2])         # series 1 at 3.5
        @test isnan(res[2][2])          # series 2 at 3.5 — cell-local NaN now propagates
        @test !isnan(res[3][2])         # series 3 at 3.5
        @test !isnan(res[2][1])         # series 2 at 2.5 — out-of-cell NaN stays hidden
    end

    @testset "Series scalar OOB ClampExtrap × deriv — cell-local per series" begin
        Y = [Float64(10i + j) for i in 1:5, j in 1:3]
        Y[5, 2] = NaN  # right boundary, series 2 only
        sitp = constant_interp(x, Series(Y); extrap = ClampExtrap())
        res = sitp(7.0; deriv = DerivOp(1))  # OOB_RIGHT
        @test !isnan(res[1])
        @test isnan(res[2])
        @test !isnan(res[3])
    end

    @testset "Series batch OOB ClampExtrap × deriv — cell-local per series" begin
        Y = [Float64(10i + j) for i in 1:5, j in 1:3]
        Y[5, 2] = NaN
        sitp = constant_interp(x, Series(Y); extrap = ClampExtrap())
        xq_batch = [7.0, 8.0]  # all OOB_RIGHT
        res = sitp(xq_batch; deriv = DerivOp(1))
        # Layout: res[k][j] — outer per series, inner per query.
        for j in eachindex(xq_batch)
            @test !isnan(res[1][j])
            @test isnan(res[2][j])
            @test !isnan(res[3][j])
        end
    end
end

# Hetero `NoInterp` axes are structurally lookup-only (no math, like Constant).
# When a non-zero `DerivOp` is requested on a `NoInterp` axis, the result is
# mathematically zero — but the existing short-circuits return `zero(Tz)` of
# the promoted type, dropping any cell-local NaN in the queried data slice.
# Closed via `data[slice] * zero(Tz)` (cell-local NaN propagation, type-promoted).
#
# Scope: covers the all-NoInterp persistent (`_eval_nointerp`, `N_r == 0`) and
# all-GridIdx oneshot (`_interp_nointerp_oneshot`, `grids_r === ()`) paths
# covering all-NoInterp (single-cell slice) and mixed Real + NoInterp (Real-axis
# kernel result × `0` if any NoInterp axis has a non-zero deriv).
@testitem "Cat G: Hetero NoInterp deriv cell-local NaN" begin
    x = collect(1.0:5.0)
    y = collect(1.0:5.0)

    @testset "All-NoInterp persistent — cell-local NaN propagates" begin
        data = [Float64(10i + j) for i in 1:5, j in 1:5]
        data[2, 3] = NaN
        itp = interp((x, y), data; method = (NoInterp(), NoInterp()))
        @test isnan(itp((GridIdx(2), GridIdx(3)); deriv = (DerivOp(1), EvalValue())))
    end

    @testset "All-NoInterp persistent — out-of-cell NaN stays hidden" begin
        data = [Float64(10i + j) for i in 1:5, j in 1:5]
        data[1, 1] = NaN
        itp = interp((x, y), data; method = (NoInterp(), NoInterp()))
        @test !isnan(itp((GridIdx(2), GridIdx(3)); deriv = (DerivOp(1), EvalValue())))
    end

    @testset "All-GridIdx oneshot — cell-local NaN propagates" begin
        data = [Float64(10i + j) for i in 1:5, j in 1:5]
        data[2, 3] = NaN
        @test isnan(
            interp(
                (x, y), data, (GridIdx(2), GridIdx(3));
                method = (NoInterp(), NoInterp()),
                deriv = (DerivOp(1), EvalValue())
            )
        )
    end

    @testset "All-GridIdx oneshot — out-of-cell NaN stays hidden" begin
        data = [Float64(10i + j) for i in 1:5, j in 1:5]
        data[1, 1] = NaN
        @test !isnan(
            interp(
                (x, y), data, (GridIdx(2), GridIdx(3));
                method = (NoInterp(), NoInterp()),
                deriv = (DerivOp(1), EvalValue())
            )
        )
    end

    @testset "Mixed Linear + NoInterp persistent — cell-local NaN propagates" begin
        # Real axis (Linear) cell (3, 4) × NoInterp axis idx 2 → data[3..4, 2]
        # NaN at data[3, 2] sits in the Real-axis cell → propagates through
        # `result * 0` when NoInterp axis carries DerivOp(1).
        data = [Float64(10i + j) for i in 1:5, j in 1:5]
        data[3, 2] = NaN
        itp = interp((x, y), data; method = (LinearInterp(), NoInterp()))
        @test isnan(itp((3.5, GridIdx(2)); deriv = (EvalValue(), DerivOp(1))))
        # Out-of-cell NaN (data[1,1] not in queried Real-axis cell [3,4]) — hidden.
        data2 = [Float64(10i + j) for i in 1:5, j in 1:5]
        data2[1, 1] = NaN
        itp2 = interp((x, y), data2; method = (LinearInterp(), NoInterp()))
        @test !isnan(itp2((3.5, GridIdx(2)); deriv = (EvalValue(), DerivOp(1))))
    end

    @testset "Mixed Linear + GridIdx oneshot — cell-local NaN propagates" begin
        data = [Float64(10i + j) for i in 1:5, j in 1:5]
        data[3, 2] = NaN
        @test isnan(
            interp(
                (x, y), data, (3.5, GridIdx(2));
                method = (LinearInterp(), NoInterp()),
                deriv = (EvalValue(), DerivOp(1))
            )
        )
        # Out-of-cell NaN stays hidden.
        data2 = [Float64(10i + j) for i in 1:5, j in 1:5]
        data2[1, 1] = NaN
        @test !isnan(
            interp(
                (x, y), data2, (3.5, GridIdx(2));
                method = (LinearInterp(), NoInterp()),
                deriv = (EvalValue(), DerivOp(1))
            )
        )
    end
end
