# ============================================================================
# PHS (polyharmonic spline) — TDD pins for KNOWN-BROKEN behaviour (PR #136)
# ============================================================================
#
# These testitems use `@test_broken` to lock down bugs and missing behaviour
# found during the PR #136 code review. Full analysis with evidence and line
# references: claudedocs/pr136_phs_code_review.md  (sections cited per pin).
#
# WHY @test_broken (recap of its 3-way semantics):
#   * expression → false   OR   throws   → recorded as Broken  (suite stays green)
#   * expression → true                  → recorded as an *Unexpected Pass* error
# So each pin is written so that it evaluates `true` ONLY once the bug is fixed.
# When a follow-up PR lands the fix, the pin turns red ("promote me to @test"):
# replace `@test_broken` with `@test` and the test becomes a permanent guard.
# Promoted pins keep their § id and drop BROKEN from the testitem name.
#
# TWO PIN SHAPES:
#   * Wrong-VALUE bugs       → `@test_broken got ≈ want`  (plain; @test_broken
#                              also swallows a *current* throw as Broken, so this
#                              works even when the value path errors today).
#   * Should-THROW-when-fixed → `@test_broken is_throwing(() -> ..., ErrType)`
#                              (the correct fixed behaviour is to raise an error;
#                              `is_throwing` lives in setup.jl / PHSBrokenHelpers).
#
# FIX-DIRECTION CAVEAT for R3 / R5 / O1 / O2:
#   These are pinned to the *conservative* fix recommended in the review —
#   REJECT the unsupported input with an `ArgumentError`. If a follow-up instead
#   chooses to *implement* the feature (true non-uniform grids, real Clamp/Wrap
#   extrapolation, derivatives of order ≥ 3), the pin will stay Broken and should
#   be REPLACED with a value test rather than promoted.
# ============================================================================

# ── R2: derivative queried EXACTLY at a grid node is silently wrong ──────────
# §R2. The d≈0 branch of the blended-gradient quotient rule drops the dominant
# `w·f′` term (w = 1 at the node), so a derivative evaluated at a grid coordinate
# is wrong while the just-off-node value is correct (a discontinuity at the node).
# NOTE: a `collect`ed Vector grid is required — a `range` grid hides the bug
# because TwicePrecision shifts the stored node coord by ~1 ulp, dodging d≈0.

@testitem "PHS BROKEN PIN §R2 — 1D first derivative at a grid node" begin
    x = collect(range(0.0, 2pi, 41))
    itp = phs_interp((x,), sin.(x))
    node = x[21]
    # today: ≈ -0.634   want (cos(node)): -1.0   — off by ~37%.
    @test_broken itp((node,); deriv = (DerivOp(1),)) ≈ cos(node) atol = 1.0e-2
end

@testitem "PHS BROKEN PIN §R2 — 2D first derivative at a grid node" begin
    gx = collect(range(0.0, 2pi, 31))
    gy = collect(range(0.0, 2pi, 31))
    data = [sin(xi) * cos(yj) for xi in gx, yj in gy]
    itp = phs_interp((gx, gy), data)
    node = (gx[15], gy[15])
    want = cos(node[1]) * cos(node[2])   # ∂/∂x of sin(x)cos(y)
    # today: ≈ 0.816   want: ≈ 0.957   — off by ~15%.
    @test_broken itp(node; deriv = (DerivOp(1), DerivOp(0))) ≈ want atol = 1.0e-2
end

# §R2 (transform path). The SAME d≈0 omission exists in the log-density transform
# kernels (_phs_eval_blended_G / _with_grad, phs_eval.jl:1039-1044 / 1294-1299),
# a DIFFERENT code path from the plain pins above (822-826). Pinned separately so
# fixing one path does not silently leave the other broken.
@testitem "PHS BROKEN PIN §R2 — log-transform first derivative at a grid node" begin
    x = collect(range(0.0, 2pi, 41))
    data = 2.0 .+ sin.(x)   # strictly positive: log-density transform domain
    itp = phs_interp((x,), data; log_reference = 1.0)
    node = x[21]
    # d/dx of (2 + sin x) = cos x. today: ≈ -0.634   want: cos(node) = -1.0.
    @test_broken itp((node,); deriv = (DerivOp(1),)) ≈ cos(node) atol = 1.0e-2
end

# NOTE — NOT pinned (latent, could not reproduce a clean failure):
# §R2 also lists the 2nd/mixed-derivative branch (phs_eval.jl:907-914) as dropping
# `sum_N1`/`sum_N1b`. A standalone failing case could not be constructed — the
# `sum_W1 ≈ 0` cancellation holds even at near-boundary 1D nodes in every tested
# config, so a @test_broken there would record an Unexpected Pass. Left documented
# in claudedocs/pr136_phs_code_review.md §R2 rather than pinned.

# ── O4: gradient / hessian / laplacian not implemented for PHS ───────────────
# §O4. PHS subtypes AbstractInterpolantND but omits _locate_cell/_eval_at_cell,
# so the vector-calculus helpers throw MethodError. README advertises them as
# supported. When implemented they must agree with the working `deriv` keyword
# path (checked here at an OFF-node point, where the deriv path is correct).

@testitem "PHS BROKEN PIN §O4 — gradient/hessian/laplacian work on PHS" begin
    gx = collect(range(0.0, 2pi, 31))
    gy = collect(range(0.0, 2pi, 31))
    data = [sin(xi) * cos(yj) for xi in gx, yj in gy]
    itp = phs_interp((gx, gy), data)
    q = (1.0, 1.0)   # off-node: deriv-keyword path is correct here

    gx_ref = itp(q; deriv = (DerivOp(1), DerivOp(0)))
    gy_ref = itp(q; deriv = (DerivOp(0), DerivOp(1)))
    d2x_ref = itp(q; deriv = (DerivOp(2), DerivOp(0)))
    d2y_ref = itp(q; deriv = (DerivOp(0), DerivOp(2)))

    # All three throw MethodError today → Broken. A real implementation agrees
    # with the deriv path (loose atol so any faithful impl flips the pin; a
    # zeros() stub would NOT agree and would correctly stay Broken).
    @test_broken (
        g = gradient(itp, q);
        length(g) == 2 &&
            isapprox(g[1], gx_ref; atol = 1.0e-2) && isapprox(g[2], gy_ref; atol = 1.0e-2)
    )
    @test_broken (
        H = hessian(itp, q);
        isapprox(H[1, 1], d2x_ref; atol = 1.0e-2) && isapprox(H[2, 2], d2y_ref; atol = 1.0e-2)
    )
    @test_broken isapprox(laplacian(itp, q), d2x_ref + d2y_ref; atol = 1.0e-2)
end

# ── O3: Complex / duck-typed value type documented but unsupported ───────────
# §O3. Tv is documented as supporting Complex, but the coeff caches and pool
# buffers are hard-typed to the grid type Tg, so evaluation throws InexactError.
# When fixed (buffers typed by promote_type(Tv,Tg)) eval returns a Complex value.

@testitem "PHS BROKEN PIN §O3 — Complex-valued data evaluates" begin
    x = collect(range(0.0, 2pi, 30))
    # Genuinely complex data (NONZERO imaginary part). A zero-imaginary Complex
    # would silently down-convert to Float64 and pass, hiding the bug — so the
    # imaginary part must carry independent information (here: cos).
    data = complex.(sin.(x), cos.(x))
    want = complex(sin(1.0), cos(1.0))
    # today: eval throws InexactError (rhs/coeff buffers hard-typed to Float64,
    # phs_eval.jl:113) → Broken. When fixed (buffers typed by promote_type) it
    # returns the interpolated complex value.
    @test_broken phs_interp((x,), data)((1.0,)) ≈ want atol = 1.0e-2
end

# ── R3: non-uniform grids accepted but evaluated as if uniform → WRONG VALUE ──
# §R3. The stencil is built from MEAN spacing while the RHS reads data at the true
# (non-uniform) node positions, so the interpolant does not even pass through its
# own data. Proven wrong-VALUE bug on a genuinely non-uniform grid:
#   node reproduction  → off by ~1-2%  (an interpolant MUST hit data at nodes)
#   linear reproduction → off by up to 0.22 at q=2.7 (want 6.4) — a degree-3 PHS
#                         reproduces linears EXACTLY (uniform control: err 6e-15).
# VALUE pins (IMPLEMENT direction — true per-node geometry). The expected values are
# closed-form: data[k] at nodes, 2q+1 everywhere for linear data. Construction is
# inside @test_broken so the pin still records Broken if a follow-up instead REJECTS
# non-uniform grids at construction (the conservative alternative in review §R3).
@testitem "PHS BROKEN PIN §R3 — non-uniform grid reproduces data and linears" begin
    # deterministic, strictly-increasing, genuinely non-uniform grid (15 nodes)
    xnu = [0.0, 0.25, 0.55, 0.7, 1.1, 1.7, 1.85, 2.4, 3.0, 3.15, 3.8, 4.5, 4.7, 5.3, 6.0]

    # (a) node reproduction: interpolant must pass through its own data.
    data = sin.(xnu) .+ 0.5
    # today: itp((xnu[12],)) ≈ -0.4998 vs data[12] ≈ -0.4775 (off ~2%).
    @test_broken phs_interp((xnu,), data; stencil_size = 6, degree = 3)((xnu[12],)) ≈ data[12] atol = 1.0e-6

    # (b) linear reproduction: degree-3 PHS is exact for linears at ANY point.
    dlin = 2.0 .* xnu .+ 1.0
    # today: itp((2.7,)) ≈ 6.62 vs 2*2.7+1 = 6.4 (off 0.22; uniform grid: err 6e-15).
    @test_broken phs_interp((xnu,), dlin; stencil_size = 6, degree = 3)((2.7,)) ≈ (2 * 2.7 + 1) atol = 1.0e-8
end

# ── R4: batch path skips domain validation ──────────────────────────────────
# §R4. _phs_batch_impl! never calls _phs_check_domain, so out-of-domain queries
# under the default NoExtrap silently return RBF-extrapolated garbage or 0.0 —
# while the scalar path correctly throws DomainError for the same query.

@testitem "PHS BROKEN PIN §R4 — batch out-of-domain query throws (NoExtrap)" setup = [PHSBrokenHelpers] begin
    x = collect(range(0.0, 2pi, 30))
    itp = phs_interp((x,), sin.(x))
    out = zeros(3)
    # 8.0 is outside [0, 2π]; scalar itp((8.0,)) throws DomainError, batch does not.
    @test_broken is_throwing(() -> itp(out, ([0.5, 1.5, 8.0],)), DomainError)
end

# ── R5: ClampExtrap / WrapExtrap accepted but never applied → WRONG VALUE ─────
# §R5. The constructor accepts any AbstractExtrap, but only NoExtrap/FillExtrap are
# implemented. This is a proven wrong-VALUE bug (boundary value cos(2π) = 1.0):
#   ClampExtrap far-OOB  (8.0) → 0.0      (should clamp to boundary 1.0)
#   ClampExtrap near-OOB (6.5) → 1.0135   (raw RBF extrap; should clamp to 1.0)
#   WrapExtrap  OOB      (8.0) → 0.0      (should return the value at the wrapped coord)
# VALUE pins (IMPLEMENT direction): each asserts the correct extrapolated value per
# the package-wide contract (verified against cubic_interp(x,y; extrap=ClampExtrap()),
# which returns 1.0). Construction is kept INSIDE @test_broken so the pin still
# records Broken — rather than erroring the testitem — if a follow-up instead takes
# the conservative REJECT route (review §R5) and throws at construction.
@testitem "PHS BROKEN PIN §R5 — Clamp/Wrap extrapolation returns the correct value" begin
    x = collect(range(0.0, 2pi, 30))
    y = cos.(x)                                  # boundary value cos(2π) = 1.0 (≠ 0)
    bnd = y[end]
    wrapped = phs_interp((x,), y)((8.0 - 2pi,))  # plain interpolant at the wrapped coord

    # ClampExtrap: every OOB query clamps to the nearest boundary value.
    @test_broken phs_interp((x,), y; extrap = ClampExtrap())((8.0,)) ≈ bnd atol = 1.0e-3
    @test_broken phs_interp((x,), y; extrap = ClampExtrap())((6.5,)) ≈ bnd atol = 1.0e-3
    # WrapExtrap: OOB query wraps into the domain and returns the in-domain value.
    @test_broken phs_interp((x,), y; extrap = WrapExtrap())((8.0,)) ≈ wrapped atol = 1.0e-6
end

# ── R5 (cont.): ExtendExtrap far-OOB collapses to 0.0 — tracking pin ──────────
# §R5. ExtendExtrap evaluates the raw interpolant past the domain. NEAR the boundary
# this works (raw RBF extension, ≈1.01, verified). But FAR out every blend weight
# w=exp(d³/(d³−a³)) has COMPACT support (exactly 0 beyond radius a), so no stencil
# contributes → silent 0.0. Returning 0.0 far-OOB is a DEFENSIBLE limitation of the
# compact-support blend, but the contract is "extend the interpolation beyond the
# domain", so a deliberate extension would be nonzero. TRACKING pin: marks the current
# silent 0.0 as the broken state; flips when far-OOB produces an explicit (finite,
# nonzero) extension. If 0.0 is later accepted as a documented limitation, delete this.
@testitem "PHS BROKEN PIN §R5 — ExtendExtrap far-OOB is a deliberate extension (not silent 0)" begin
    x = collect(range(0.0, 2pi, 30))
    y = cos.(x)
    v = phs_interp((x,), y; extrap = ExtendExtrap())((8.0,))   # today: 0.0 (blend collapse)
    @test_broken isfinite(v) && !iszero(v)
end

# ── O1: derivative order ≥ 3 silently returns 0.0 — should return the TRUE value ─
# §O1. Unlike a cubic spline (a piecewise degree-3 polynomial whose 4th+ derivatives
# genuinely vanish), a degree-3 PHS is transcendental: the exponential blend weight
# w(d)=exp(d³/(d³−a³)) makes derivatives of EVERY order nonzero (kernel r³ alone gives
# nonzero d3; the blend additionally gives nonzero d4, d5, …). The code's
# `total_deriv ≥ 3 → return zero` wrongly treats PHS as a polynomial.
# VALUE pins (IMPLEMENT direction): the analytic high-order derivative must match a
# finite-difference reference built from the (correct) 2nd-derivative path. NOTE the
# correct value is NONZERO but not necessarily positive — d4 ≈ -5.34 here. q=3.3 gives
# a large, FD-stable reference (d3 ≈ 1.34, d4 ≈ -5.34; both stable to <0.1% under h).
@testitem "PHS BROKEN PIN §O1 — derivative order ≥ 3 returns the true nonzero value" begin
    x = collect(range(0.0, 2pi, 30))
    itp = phs_interp((x,), sin.(x))
    q = 3.3
    h = 1.0e-3
    d2(t) = itp((t,); deriv = (DerivOp(2),))           # 2nd-deriv path is correct off-node
    d3_ref = (d2(q + h) - d2(q - h)) / (2h)             # ≈ 1.34  (true 3rd derivative)
    d4_ref = (d2(q + h) - 2 * d2(q) + d2(q - h)) / h^2  # ≈ -5.34 (true 4th derivative)

    # today both return 0.0 (silent bug); a correct implementation matches the FD reference.
    @test_broken itp((q,); deriv = (DerivOp(3),)) ≈ d3_ref rtol = 0.05
    @test_broken itp((q,); deriv = (DerivOp(4),)) ≈ d4_ref rtol = 0.05
end

# ── O2: blend_factor not validated → silent all-zero output ──────────────────
# §O2. blend_factor ≤ 0 makes every blend weight vanish so the interpolant
# returns 0.0 everywhere; degree and stencil_size are validated but this is not.

@testitem "PHS BROKEN PIN §O2 — invalid blend_factor rejected" setup = [PHSBrokenHelpers] begin
    x = collect(range(0.0, 2pi, 41))
    y = sin.(x)
    @test_broken is_throwing(() -> phs_interp((x,), y; blend_factor = -1.0), ArgumentError)
    @test_broken is_throwing(() -> phs_interp((x,), y; blend_factor = 0.0), ArgumentError)
end

# ── F4/O4 (fixed): the log transform rejects non-positive data ────────────────
# §F4 (and the phs.md "Custom Reference" example). Under the log transform the
# constructor stores `log(data/ρ₀)`. Negative data used to throw DomainError and an
# exact zero stored log(0) = -Inf silently, so evaluation near that node returned
# NaN. The constructor now checks every ratio and throws ArgumentError. (The data
# 1+cos is ≥ 0 and hits exactly 0 at x = π.)
@testitem "PHS PIN §F4 — log transform rejects non-positive (zero) data" setup = [PHSBrokenHelpers] begin
    x = collect(range(0.0, 2pi, 41))
    data = 1.0 .+ cos.(x)   # ≥ 0 with an exact zero at x = π; no negatives
    @test is_throwing(() -> phs_interp((x,), data; log_reference = 1.0), ArgumentError)
end

# ── F3: 1D bare-vector construction convenience missing ──────────────────────
# §F3. Other families accept the bare 1D form (e.g. cubic_interp(x, y)); PHS only
# accepts the 1-tuple form phs_interp((x,), y), so phs_interp(x, y) is a MethodError.
# This is an IMPLEMENT-direction pin (unlike the reject-direction throw pins): a
# follow-up wrapper `phs_interp(x::AbstractVector, y::AbstractVector; ...)` should
# delegate to the tuple form, so the two must agree exactly.
@testitem "PHS BROKEN PIN §F3 — 1D bare-vector construction works" begin
    x = collect(range(0.0, 2pi, 30))
    y = sin.(x)
    want = phs_interp((x,), y)((1.0,))   # canonical tuple form
    # today: phs_interp(x, y) → MethodError (swallowed as Broken). When the wrapper
    # lands it returns an equivalent interpolant agreeing with the tuple form.
    @test_broken phs_interp(x, y)((1.0,)) ≈ want atol = 1.0e-12
end

# ============================================================================
# Regressions from the blend-default and weighted-node-selection changes, and
# gaps against the ND query surfaces. Sections cite the same review document.
# ============================================================================

# ── R7: one-shot and persistent constructors disagree on the default blend ────
# §R7. The persistent constructor defaults to blend_factor = 1.0, while the three
# one-shot `phs_interp(grids, data, queries)` methods still default to 2.0, so the
# same inputs give different results depending on the entry point. Fixed when both
# paths resolve the default from one place.
@testitem "PHS BROKEN PIN §R7 — one-shot and persistent share the default blend_factor" begin
    x = range(0.0, π, 15)
    y = range(0.0, π, 15)
    data = [sin(xi + yj) for xi in x, yj in y]
    q = (1.0, 1.5)
    # today: ≈ 0.59862 (persistent) vs ≈ 0.59853 (one-shot) — off by ~1e-4.
    @test_broken phs_interp((x, y), data; stencil_size = 5)(q) ≈
        phs_interp((x, y), data, q; stencil_size = 5) atol = 1.0e-12
end

# ── R8: default blend radius leaves 4-D cell centres uncovered → exact 0.0 ─────
# §R8. The blend weight vanishes at distance a = blend_factor · h. A cell centre is
# (h/2)·√N from its nearest nodes, so with the default blend_factor = 1.0 all
# weights are ~0 near 4-D (and higher) cell centres, Σw < eps, and the evaluator
# returns zero. blend_factor = 2.0 (the paper's a = 2 × longest grid step) gives
# 3.797 here. Fixed by a dimension-aware default.
@testitem "PHS BROKEN PIN §R8 — 4-D default blend_factor covers cell centres" begin
    x = range(0.0, 3.0, 7)
    h = step(x)
    f(p) = sum(sin, p)
    data = [f(Tuple(I) .* h .- h) for I in CartesianIndices(ntuple(_ -> 7, 4))]
    c = ntuple(_ -> (x[3] + x[4]) / 2, 4)
    # today: exactly 0.0   want: ≈ 3.796
    @test_broken phs_interp(ntuple(_ -> x, 4), data; stencil_size = 3)(c) ≈ f(c) rtol = 1.0e-2
end

# ── R9: fixed 27-slot blend buffer overflows for wider blends ─────────────────
# §R9. `_phs_eval_blended_G_with_hess` (log transform + 2nd derivatives) collects
# blend nodes into a buffer of fixed length 27, the candidate count for 3-D with
# blend_factor ≤ 1. Wider blends (≳ 1.8 in 3-D, including the paper's 2.0 and the
# one-shot default) overflow it with a BoundsError. With bounds checks disabled the
# same write goes out of bounds silently, so the evaluation is skipped there (CI
# always runs with bounds checks on).
@testitem "PHS BROKEN PIN §R9 — log-transform 2nd derivative with blend_factor = 2.0" begin
    x = range(0.0, 3.0, 13)
    data = [exp(-((a - 1.5)^2 + (b - 1.5)^2 + (c - 1.5)^2)) + 0.5 for a in x, b in x, c in x]
    q = (1.3, 1.37, 1.21)
    want = (4 * (q[1] - 1.5)^2 - 2) * exp(-sum(abs2, q .- 1.5))   # ∂²/∂x² ≈ -1.598
    ops = (DerivOp(2), DerivOp(0), DerivOp(0))
    if Base.JLOptions().check_bounds == 2
        @test_broken false
        @test_broken false
    else
        # today: BoundsError (swallowed as Broken)   want: ≈ -1.598 (-1.615 before the regression)
        itp = phs_interp((x, x, x), data; stencil_size = 4, blend_factor = 2.0, log_reference = 1.0)
        @test_broken itp(q; deriv = ops) ≈ want rtol = 5.0e-2
        # the one-shot default is 2.0, so the default call overflows as well
        @test_broken phs_interp((x, x, x), data, q; stencil_size = 4, log_reference = 1.0, deriv = ops) ≈
            want rtol = 5.0e-2
    end
end

# ── R10: truncated blend makes log-transform 2nd derivatives discontinuous ─────
# §R10. The same `_phs_eval_blended_G_with_hess` keeps only the 7 heaviest blend
# nodes and stops once 90% of the total weight is accumulated. The kept set flips
# where two nodes tie in weight (cell mid-planes), so the second derivative jumps
# there even with the default blend_factor. The other blend paths sum every
# candidate and stay continuous. The probe crosses the mid-plane x = 1.125.
@testitem "PHS BROKEN PIN §R10 — log-transform 2nd derivative is continuous" begin
    x = range(0.0, 3.0, 13)
    data = [exp(-((a - 1.5)^2 + (b - 1.5)^2 + (c - 1.5)^2)) + 0.5 for a in x, b in x, c in x]
    ops = (DerivOp(2), DerivOp(0), DerivOp(0))
    ts = range(1.123, 1.127; step = 1.0e-6)
    maxjump(itp) = maximum(abs, diff([itp((t, 1.37, 1.21); deriv = ops) for t in ts]))
    itp = phs_interp((x, x, x), data; stencil_size = 4, log_reference = 1.0)
    # today: 0.154, a step that does not shrink with finer sampling   want: ~6e-5
    @test_broken maxjump(itp) < 1.0e-3
    # control: the same quantity without the log transform is continuous today
    @test maxjump(phs_interp((x, x, x), data; stencil_size = 4)) < 1.0e-3
end

# ── O2 (coverage): blend_factor below the cell-coverage threshold → silent 0.0 ─
# §O2. Coverage needs a > (h/2)·√N, since a cell centre is the point farthest from
# every node. Below that, queries near cell centres get Σw < eps and evaluate to
# 0.0: 1-D with blend_factor = 0.3 at a cell midpoint, or 4-D with an explicit 1.0.
# Pinned to the conservative fix (reject); replace with value pins if a follow-up
# makes such blends work instead.
@testitem "PHS BROKEN PIN §O2 — blend_factor below the coverage threshold rejected" setup = [PHSBrokenHelpers] begin
    x1 = collect(range(0.0, 2pi, 41))
    @test_broken is_throwing(() -> phs_interp((x1,), sin.(x1); blend_factor = 0.3), ArgumentError)
    x4 = range(0.0, 3.0, 7)
    data4 = [sum(sin, Tuple(I) .* step(x4) .- step(x4)) for I in CartesianIndices(ntuple(_ -> 7, 4))]
    @test_broken is_throwing(
        () -> phs_interp(ntuple(_ -> x4, 4), data4; stencil_size = 3, blend_factor = 1.0),
        ArgumentError,
    )
end

# ── O9: PHS does not take part in the ND query surfaces ───────────────────────
# §O9. An N ≥ 2 GriddedQuery needs the gridded protocol (`_sample_data`), and
# GridIdx needs query resolution (`_phs_check_domain` / `_phs_eval` reject it);
# both are MethodErrors today. The N = 1 GriddedQuery forwards already work.
@testitem "PHS BROKEN PIN §O9 — GriddedQuery (N ≥ 2) and GridIdx queries" begin
    x = range(0.0, 1.0, 11)
    y = range(0.0, 1.0, 11)
    data = [sin(a) * cos(b) for a in x, b in y]
    itp = phs_interp((x, y), data; stencil_size = 4)
    ax1 = range(0.1, 0.9, 5)
    ax2 = range(0.1, 0.9, 4)
    @test_broken itp(GriddedQuery((ax1, ax2))) ≈ [itp((a, b)) for a in ax1, b in ax2]
    yq = 0.37
    @test_broken itp((GridIdx(4), yq)) ≈ itp((x[4], yq))
    @test_broken itp(([GridIdx(4), GridIdx(5)], [yq, yq])) ≈ [itp((x[4], yq)), itp((x[5], yq))]
end

# ── F5 (fixed): the reference_data keyword is gone ─────────────────────────────
# §F5. `reference_data` was silently dropped unless `reference_interp` was also
# given. Both keywords are replaced by `log_reference`, so the old keyword is now
# rejected outright.
@testitem "PHS PIN §F5 — the removed reference_data keyword is rejected" begin
    x = collect(range(0.0, 2pi, 41))
    data = 2.0 .+ sin.(x)
    @test_throws MethodError phs_interp((x,), data; reference_data = ones(length(x)))
end

# ── R11 (fixed): the constant reference answers an order-0 deriv tuple ──────────
# §R11 (claudedocs/design/2026-09-30-phs-log-reference-api.md §4.3). `ConstantRef`
# returned zero whenever `deriv !== nothing`, so an all-`EvalValue` tuple got 0
# instead of the value. The internal constant reference decides by the total order.
@testitem "PHS PIN §R11 — constant reference answers an order-0 deriv tuple with its value" begin
    q = (0.1, 0.2)
    @test FastInterpolations._phs_ref_value(2.5, q) == 2.5
    @test FastInterpolations._phs_ref_deriv(2.5, q, (DerivOp(1), EvalValue())) == 0.0
    @test FastInterpolations._phs_ref_deriv(2.5, q, (EvalValue(), EvalValue())) == 2.5
    @test FastInterpolations._phs_ref_deriv(5, q, (DerivOp(1), EvalValue())) === 0
end

# ── F6: value-only reference fails derivative queries with a MethodError ──────
# §F6 (design note §2.5). A plain function `p -> ρ₀(p)` answers value queries, but a
# derivative query calls `ρ₀(q; deriv = ops)` and fails deep inside with
# `MethodError: no method matching (::var"#…")(…; deriv)`. Fixed when such a
# reference is recognised at build and derivative queries raise a clear ArgumentError.
@testitem "PHS BROKEN PIN §F6 — value-only reference: derivative query raises ArgumentError" setup = [PHSBrokenHelpers] begin
    x = range(0.0, 1.0; length = 12)
    y = range(0.0, 1.0; length = 12)
    rho = [exp(-xi) * (1.5 + yj) for xi in x, yj in y]
    itp = phs_interp((x, y), rho; log_reference = p -> exp(-p[1]))
    q = (0.43, 0.61)
    @test itp(q) ≈ exp(-q[1]) * (1.5 + q[2]) rtol = 1.0e-3   # control: value path works
    # today: MethodError from the reference, which takes no `deriv` keyword
    @test_broken is_throwing(() -> itp(q; deriv = (DerivOp(1), EvalValue())), ArgumentError)
end

# ── F7 (fixed): an array passed as the reference is rejected clearly ────────────
# §F7 (design note §2.5, §3.3). A same-shape ρ₀ array is not callable and used to
# leak `MethodError: objects of type Matrix{Float64} are not callable`. Arrays are
# now rejected with an explanatory ArgumentError: ρ₀ is needed between nodes too,
# and a reference sampled on the data grid adds no information.
@testitem "PHS PIN §F7 — array reference rejected with ArgumentError" setup = [PHSBrokenHelpers] begin
    x = range(0.0, 1.0; length = 12)
    y = range(0.0, 1.0; length = 12)
    rho = [exp(-xi) * (1.5 + yj) for xi in x, yj in y]
    rho0 = [exp(-xi) for xi in x, yj in y]
    @test is_throwing(() -> phs_interp((x, y), rho; log_reference = rho0), ArgumentError)
end
