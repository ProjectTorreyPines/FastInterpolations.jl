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
    itp = phs_interp((x,), data; reference_interp = ConstantRef(1.0))
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
    @test_broken (g = gradient(itp, q);
        length(g) == 2 &&
            isapprox(g[1], gx_ref; atol = 1.0e-2) && isapprox(g[2], gy_ref; atol = 1.0e-2))
    @test_broken (H = hessian(itp, q);
        isapprox(H[1, 1], d2x_ref; atol = 1.0e-2) && isapprox(H[2, 2], d2y_ref; atol = 1.0e-2))
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

# ── F4/O4: log-density transform silently accepts non-positive data ──────────
# §F4 (and the phs.md "Custom Reference" example). Under the log transform the
# constructor stores `log(data/ρ₀)`. NEGATIVE data already throws DomainError
# (loud, fine), but an EXACT ZERO yields log(0) = -Inf silently — construction
# succeeds and evaluation near that node returns NaN. The fix should require
# strictly-positive data and reject zeros too. (Pin uses 1+cos, which is ≥ 0 and
# hits exactly 0 at x = π — no negatives, so today it does NOT throw.)
@testitem "PHS BROKEN PIN §F4 — log transform rejects non-positive (zero) data" setup = [PHSBrokenHelpers] begin
    x = collect(range(0.0, 2pi, 41))
    data = 1.0 .+ cos.(x)   # ≥ 0 with an exact zero at x = π; no negatives
    @test_broken is_throwing(
        () -> phs_interp((x,), data; reference_interp = ConstantRef(1.0)),
        Union{DomainError, ArgumentError},
    )
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
