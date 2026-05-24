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
#   Cat F  — Constant 1D right-edge + Series cell-local NaN (`0 * first(y)` → cell-local idx)
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

# OOB ClampExtrap/FillExtrap × deriv routes through
# `_eval_extrapolation(::DerivOp, …)` → `_promote_extrap_zero(y_bnd, xq)`.
# The `Number` overload currently strips `val` via `zero(xq) * zero(val)`
# (type-only zero) while the `AbstractArray` overload uses `0 .* val .+ …`
# (NaN propagates element-wise). The asymmetry is the bug — boundary NaN
# disappears at OOB deriv queries for scalar `Tv`, but propagates for
# Array `Tv`. The fix mirrors the Array form on the Number form
# (`0 * val + zero(xq) * zero(val)`), unifying cell-local boundary-data
# carrier behavior under OOB deriv queries.
#
# Scope: only the chosen-side boundary's NaN must propagate (e.g.,
# `y[1] = NaN` with OOB_LEFT query → NaN; same `y` with OOB_RIGHT query
# → finite). Out-of-cell NaN must NOT leak through OOB extrap.
@testitem "Cat E: OOB cell-local NaN through _promote_extrap_zero (Number case)" begin
    using ForwardDiff
    using FastInterpolations: _promote_extrap_zero

    @testset "_promote_extrap_zero helper — Number vs Array consistency" begin
        # Array case already propagates NaN element-wise — sanity baseline.
        # `@inferred` pins type stability against future regressions on the
        # OOB extrap helper path (Cat A only covers in-domain).
        res_arr = @inferred _promote_extrap_zero([NaN, 1.0], 0.5)
        @test isnan(res_arr[1])

        # Number case must match: NaN at boundary propagates as NaN result.
        @test (@inferred _promote_extrap_zero(NaN, 0.5)) isa Float64
        @test isnan(_promote_extrap_zero(NaN, 0.5))

        # Carrier preserved + primal NaN propagated through Dual xq.
        xq_d = ForwardDiff.Dual{Nothing}(0.5, 1.0)
        res_d = @inferred _promote_extrap_zero(NaN, xq_d)
        @test res_d isa typeof(xq_d)
        @test isnan(ForwardDiff.value(res_d))
    end

    @testset "1D OOB ClampExtrap × deriv: queried-boundary NaN propagates" begin
        x = collect(1.0:5.0)
        y_nan_left = [NaN, 20.0, 30.0, 40.0, 50.0]
        y_nan_right = [10.0, 20.0, 30.0, 40.0, NaN]
        xq_oob_lo, xq_oob_hi = -1.0, 7.0

        for method in (linear_interp, constant_interp, pchip_interp, cardinal_interp, akima_interp)
            @test isnan(method(x, y_nan_left, xq_oob_lo; extrap = ClampExtrap(), deriv = DerivOp(1)))
            @test isnan(method(x, y_nan_right, xq_oob_hi; extrap = ClampExtrap(), deriv = DerivOp(1)))
            # Cell-local: NaN at the OTHER boundary does NOT propagate.
            @test !isnan(method(x, y_nan_left, xq_oob_hi; extrap = ClampExtrap(), deriv = DerivOp(1)))
            @test !isnan(method(x, y_nan_right, xq_oob_lo; extrap = ClampExtrap(), deriv = DerivOp(1)))
        end
    end

    @testset "1D OOB FillExtrap × deriv: queried-boundary NaN propagates" begin
        # FillExtrap routes y_bnd (boundary data, not fill_value) into
        # `_promote_extrap_zero` for deriv. Consistent with Array case behavior.
        x = collect(1.0:5.0)
        y_nan_left = [NaN, 20.0, 30.0, 40.0, 50.0]
        y_nan_right = [10.0, 20.0, 30.0, 40.0, NaN]
        xq_oob_lo, xq_oob_hi = -1.0, 7.0

        for method in (linear_interp, constant_interp, pchip_interp, cardinal_interp, akima_interp)
            @test isnan(method(x, y_nan_left, xq_oob_lo; extrap = FillExtrap(0.0), deriv = DerivOp(1)))
            @test isnan(method(x, y_nan_right, xq_oob_hi; extrap = FillExtrap(0.0), deriv = DerivOp(1)))
            @test !isnan(method(x, y_nan_left, xq_oob_hi; extrap = FillExtrap(0.0), deriv = DerivOp(1)))
            @test !isnan(method(x, y_nan_right, xq_oob_lo; extrap = FillExtrap(0.0), deriv = DerivOp(1)))
        end
    end
end

# Constant 1D `_constant_eval_at_point` / `_constant_eval_at_anchor` short-
# circuit at `xi == last(x)` (right-edge seam) and Constant Series boundary
# / in-domain deriv branches use `0 * first(y)` for the deriv-zero result —
# not cell-local: NaN at `y[1]` leaks even when the queried cell is the
# right-edge cell `(y[end-1], y[end])`. The EvalValue sibling branches
# already use cell-local `y[idxR]` / `y[n_pts, k]` / etc. — fixing the
# deriv branches to mirror that pattern restores the cell-local NaN
# invariant established by Phase 3 + Cat E.
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

    @testset "Series scalar — right-boundary cell-local per series" begin
        # Series matrix layout: y[time_idx, series_k]. NaN at right-boundary of
        # ONE series → only that series's result is NaN; other series unaffected.
        Y = [Float64(10i + j) for i in 1:5, j in 1:3]
        Y[5, 2] = NaN  # right boundary, series 2 only
        sitp = constant_interp(x, Series(Y))
        res = sitp(xi_edge; deriv = DerivOp(1))
        @test !isnan(res[1])
        @test isnan(res[2])
        @test !isnan(res[3])
    end

    @testset "Series batch — in-domain cell-local per series" begin
        # `_eval_constant_series_anchored` line 627 deriv branch uses
        # `0 * first(y)` — strips series-k. Batch path hits this; scalar
        # path takes a different per-series-aware route already.
        Y = [Float64(10i + j) for i in 1:5, j in 1:3]
        Y[3, 2] = NaN  # interior cell corner, series 2 only
        sitp = constant_interp(x, Series(Y))
        xq_batch = [2.5, 3.5]  # 3.5 hits cell (3, 4); 2.5 hits cell (2, 3)
        res = sitp(xq_batch; deriv = DerivOp(1))
        # Layout: res[k][j] — outer per series, inner per query.
        @test !isnan(res[1][2])     # series 1 at 3.5
        @test isnan(res[2][2])      # series 2 at 3.5 — cell-local NaN propagates
        @test !isnan(res[3][2])     # series 3 at 3.5
        @test !isnan(res[2][1])     # series 2 at 2.5 — out-of-cell NaN stays hidden
    end

    @testset "Series scalar OOB ClampExtrap × deriv — cell-local per series" begin
        # `_constant_extrap_boundary_value` deriv overload uses `0 * first(y)`
        # — strips series-k AND drops NaN at boundary of non-first series.
        Y = [Float64(10i + j) for i in 1:5, j in 1:3]
        Y[5, 2] = NaN  # right boundary, series 2 only
        sitp = constant_interp(x, Series(Y); extrap = ClampExtrap())
        res = sitp(7.0; deriv = DerivOp(1))  # OOB_RIGHT
        @test !isnan(res[1])
        @test isnan(res[2])
        @test !isnan(res[3])
    end

    @testset "Series batch OOB ClampExtrap × deriv — cell-local per series" begin
        # `_fill_constant_extrap_simd!` deriv overload fills uniformly with
        # `0 * first(y)` — same per-series leak as the scalar variant.
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
# This mirrors the Constant Phase 3 bug and is closed via `data[slice] * zero(Tz)`
# (cell-local NaN propagation, type-promoted).
#
# Scope: covers the all-NoInterp persistent (`_eval_nointerp`, `N_r == 0`) and
# all-GridIdx oneshot (`_interp_nointerp_oneshot`, `grids_r === ()`) paths
# where the data is fully sliced to a single cell. The mixed case
# (`_interp_nointerp_oneshot` line ~489 with Real axes still present) is a
# known scope exception — strict cell-local requires running the Real-axis
# interp first, deferred as a separate follow-up.
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
end
