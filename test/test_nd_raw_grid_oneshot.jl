# ============================================================================
# ND one-shot raw-grid evaluation (no eager grid conversion)
# ============================================================================
#
# Linear/Constant ND one-shot search & evaluate the RAW grid tuple (via the
# kernel's own `map(_resolve_axis, grids, bcs)` + promote-compare search),
# instead of eagerly materialising `Tg.(x)` for every non-`Tg` Vector axis.
#
# Contract:
#   - Int / Float64 Vector axes: zero heap alloc on the scalar one-shot,
#     bit-identical to the Float64-grid / persistent-interpolant result.
#   - Heterogeneous axes (e.g. Int × Float64) stay type-stable (::Tr guard).
#   - ForwardDiff through the query is unchanged.
#   - Float32 Vector axes: within a few ULP of the Float64 reference.

@testitem "ND one-shot raw-grid eval (no eager grid conversion)" setup = [AllocConstants] begin
    using ForwardDiff

    # ---- zero-alloc scalar one-shot on Int Vector grids (the migration) ----
    # Function barriers: setup + warmup + @allocated inside one function to
    # avoid @testset-scope boxing artifacts (mirrors test_nd_constant.jl).
    function _alloc_linear_int_vec_2d()
        x = [0, 1, 2, 3, 4]      # Int Vector axis
        y = [0, 1, 2, 3]
        data = [2.0 * xi + 3.0 * yj for xi in x, yj in y]
        q = (1.5, 2.5)
        linear_interp((x, y), data, q)
        linear_interp((x, y), data, q)
        @allocated linear_interp((x, y), data, q)
    end
    function _alloc_constant_int_vec_2d()
        x = [0, 1, 2, 3, 4]
        y = [0, 1, 2, 3]
        data = [2.0 * xi + 3.0 * yj for xi in x, yj in y]
        q = (1.5, 2.5)
        constant_interp((x, y), data, q)
        constant_interp((x, y), data, q)
        @allocated constant_interp((x, y), data, q)
    end
    function _alloc_linear_int_vec_3d()
        x = [0, 1, 2, 3]
        y = [0, 1, 2]
        z = [0, 1, 2, 3, 4]
        data = [xi + yj + zk for xi in x, yj in y, zk in z]
        q = (1.5, 1.5, 2.5)
        linear_interp((x, y, z), data, q)
        linear_interp((x, y, z), data, q)
        @allocated linear_interp((x, y, z), data, q)
    end

    @testset "zero-alloc scalar one-shot on Int Vector grids" begin
        @test _alloc_linear_int_vec_2d() <= ND_ALLOC_THRESHOLD
        @test _alloc_constant_int_vec_2d() <= ND_ALLOC_THRESHOLD
        @test _alloc_linear_int_vec_3d() <= ND_ALLOC_THRESHOLD
    end

    # ---- bit-identical: raw Int grid eval === Float64 grid eval ----
    @testset "bit-identical: Int Vector grid === Float64 grid" begin
        x = [0, 1, 2, 3, 4]
        y = [0, 1, 2, 3]
        data = [2.0 * xi + 3.0 * yj - 0.5 * xi * yj for xi in x, yj in y]
        xf = Float64.(x)
        yf = Float64.(y)
        for q in [(1.5, 2.5), (0.3, 0.7), (3.9, 0.1), (2.0, 3.0)]
            @test linear_interp((x, y), data, q) === linear_interp((xf, yf), data, q)
            @test constant_interp((x, y), data, q) === constant_interp((xf, yf), data, q)
        end
    end

    # ---- one-shot === persistent interpolant (eager-convert path, unchanged) ----
    @testset "one-shot === persistent interpolant" begin
        x = [0, 1, 2, 3, 4]
        y = [0, 1, 2, 3]
        data = [sin(1.0 * xi) + cos(1.0 * yj) for xi in x, yj in y]
        itp_lin = linear_interp((x, y), data)
        itp_con = constant_interp((x, y), data)
        for q in [(1.5, 2.5), (0.3, 0.7), (3.9, 0.1)]
            @test linear_interp((x, y), data, q) === itp_lin(q)
            @test constant_interp((x, y), data, q) === itp_con(q)
        end
    end

    # ---- type-stable (::Tr boxing guard holds on raw / heterogeneous axes) ----
    @testset "type-stable (no boxing) on raw / heterogeneous axes" begin
        x_int = [0, 1, 2, 3, 4]
        y_int = [0, 1, 2, 3]
        y_f64 = Float64.(y_int)
        data = [2.0 * xi + 3.0 * yj for xi in x_int, yj in y_int]
        data_int = [2 * xi + 3 * yj for xi in x_int, yj in y_int]   # Int data → Tr=Float64
        q = (1.5, 2.5)
        @test (@inferred linear_interp((x_int, y_int), data, q)) isa Float64
        @test (@inferred linear_interp((x_int, y_f64), data, q)) isa Float64   # heterogeneous
        @test (@inferred linear_interp((x_int, y_int), data_int, q)) isa Float64  # Int data
        @test (@inferred constant_interp((x_int, y_int), data, q)) isa Float64
    end

    # ---- heterogeneous (Int × Float64) axes: value correct ----
    @testset "heterogeneous (Int × Float64) axes value" begin
        x = [0, 1, 2, 3, 4]          # Int
        y = [0.0, 0.5, 1.0, 1.5]     # Float64
        f(xi, yj) = 2.0 * xi + 3.0 * yj + 1.0
        data = [f(xi, yj) for xi in x, yj in y]
        for q in [(1.5, 0.75), (3.2, 1.1)]
            @test linear_interp((x, y), data, q) ≈ f(q[1], q[2]) atol = 1.0e-12
        end
    end

    # ---- ForwardDiff through the ND query on an Int Vector grid ----
    @testset "ForwardDiff through ND query on Int Vector grid" begin
        x = [0, 1, 2, 3, 4]
        y = [0, 1, 2, 3]
        a, b, c = 2.0, 3.0, -0.5
        data = [a * xi + b * yj + c * xi * yj for xi in x, yj in y]
        # Within a single cell the bilinear surface has analytic partials
        # ∂/∂x = a + c·y, ∂/∂y = b + c·x.
        p = [1.5, 2.5]
        g = ForwardDiff.gradient(pp -> linear_interp((x, y), data, (pp[1], pp[2])), p)
        @test g[1] ≈ a + c * p[2] atol = 1.0e-10
        @test g[2] ≈ b + c * p[1] atol = 1.0e-10
    end

    # ---- Float32 Vector axis: within a few ULP of the Float64 reference ----
    @testset "Float32 Vector axis ≈ Float64 reference (relaxed bar)" begin
        xf = Float32[0, 1, 2, 3, 4]
        yf = Float32[0, 1, 2, 3]
        x64 = Float64.(xf)
        y64 = Float64.(yf)
        data = [2.0 * xi + 3.0 * yj for xi in xf, yj in yf]
        for q in [(1.5, 2.5), (0.3, 0.7)]
            ref = linear_interp((x64, y64), data, q)
            got = linear_interp((xf, yf), data, q)
            @test got ≈ ref rtol = 1.0e-6
        end
    end
end

# ============================================================================
# Cubic ND one-shot — raw Int grid hits the (Float) per-axis cache after warmup
# ============================================================================
#
# Cubic ND builds per-axis spline caches via `_get_cubic_cache`, which memoises
# by grid object id. The eager `_nd_promote_grids` built a fresh `Tg.(x)` Vector
# every call (new id → permanent cache miss + the conversion alloc). Passing the
# RAW grid (stable id) lets the cache hit, so warm scalar one-shot on an Int grid
# is zero-alloc — matching 1D cubic. (Batch keeps eager-convert; see linear note.)

@testitem "Cubic ND one-shot raw-grid (warm cache hit, no eager convert)" setup = [AllocConstants] begin
    using ForwardDiff

    # ---- warm zero-alloc on Int Vector grids (default OnTheFly + PreCompute) ----
    # Function barrier: build grid once, warm the per-axis cache, then @allocated.
    function _alloc_cubic_nd_int_otf_2d()
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [0, 1, 2, 3, 4, 5]
        data = [sin(1.0 * a) + cos(1.0 * b) for a in x, b in y]
        q = (3.4, 2.6)
        for _ in 1:3
            cubic_interp((x, y), data, q)
        end
        @allocated cubic_interp((x, y), data, q)
    end
    # PreCompute (the practically-used cubic strategy): axes go through the POOLED
    # value-matched wrap (`_cache_axes_pooled` — acquire + copyto!, no heap `Tg.(x)`),
    # and the spline autocache hits via its content-match pass (pooled buffers don't
    # keep a stable objectid) — so the warm Int-grid one-shot stays zero-alloc.
    function _alloc_cubic_nd_int_pre_2d()
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [0, 1, 2, 3, 4, 5]
        data = [sin(1.0 * a) + cos(1.0 * b) for a in x, b in y]
        q = (3.4, 2.6)
        for _ in 1:3
            cubic_interp((x, y), data, q; coeffs = PreCompute())
        end
        @allocated cubic_interp((x, y), data, q; coeffs = PreCompute())
    end
    function _alloc_cubic_nd_int_otf_3d()
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4]
        z = [0, 1, 2, 3, 4, 5, 6]
        data = [a + 0.5b + 0.25c for a in x, b in y, c in z]
        q = (2.4, 1.6, 3.8)
        for _ in 1:3
            cubic_interp((x, y, z), data, q)
        end
        @allocated cubic_interp((x, y, z), data, q)
    end

    @testset "warm zero-alloc scalar one-shot on Int Vector grids" begin
        @test _alloc_cubic_nd_int_otf_2d() <= ND_ALLOC_THRESHOLD
        @test _alloc_cubic_nd_int_otf_3d() <= ND_ALLOC_THRESHOLD
        @test _alloc_cubic_nd_int_pre_2d() <= ND_ALLOC_THRESHOLD   # PreCompute now raw
    end

    # ---- bit-identical: raw Int grid === Float64 grid (value) ----
    @testset "Int Vector grid === Float64 grid" begin
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [0, 1, 2, 3, 4, 5]
        data = [sin(1.0 * a) + cos(1.0 * b) for a in x, b in y]
        xf = Float64.(x)
        yf = Float64.(y)
        for q in [(3.4, 2.6), (0.3, 0.7), (6.9, 4.1)]
            @test cubic_interp((x, y), data, q) === cubic_interp((xf, yf), data, q)
            @test cubic_interp((x, y), data, q; coeffs = PreCompute()) ===
                cubic_interp((xf, yf), data, q; coeffs = PreCompute())
        end
    end

    # ---- one-shot ≈ persistent interpolant ----
    # NB: `≈` not `===` — the default scalar one-shot is OnTheFly (sequential
    # collapse) while the persistent interpolant is PreCompute (full tensor
    # partials); the two algorithms agree to ~1 ULP, not bit-for-bit.
    @testset "one-shot ≈ persistent CubicInterpolantND" begin
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [0, 1, 2, 3, 4, 5]
        data = [sin(1.0 * a) + cos(1.0 * b) for a in x, b in y]
        itp = cubic_interp((x, y), data)
        for q in [(3.4, 2.6), (0.3, 0.7), (6.9, 4.1)]
            @test cubic_interp((x, y), data, q) ≈ itp(q)
        end
    end

    # ---- type stability (::Tr) on raw / heterogeneous axes ----
    @testset "type-stable (::Tr) on raw / heterogeneous axes" begin
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        yi = [0, 1, 2, 3, 4, 5]
        yf = Float64.(yi)
        data = [sin(1.0 * a) + cos(1.0 * b) for a in x, b in yi]
        q = (3.4, 2.6)
        @test (@inferred cubic_interp((x, yi), data, q)) isa Float64
        @test (@inferred cubic_interp((x, yf), data, q)) isa Float64   # heterogeneous
    end

    # ---- ForwardDiff through the ND query on an Int Vector grid ----
    @testset "ForwardDiff through ND query on Int Vector grid" begin
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [0, 1, 2, 3, 4, 5]
        data = [1.0 * a^2 + 2.0 * b for a in x, b in y]
        p = [3.4, 2.6]
        g = ForwardDiff.gradient(pp -> cubic_interp((x, y), data, (pp[1], pp[2])), p)
        gf = ForwardDiff.gradient(pp -> cubic_interp((Float64.(x), Float64.(y)), data, (pp[1], pp[2])), p)
        @test g ≈ gf atol = 1.0e-10
    end
end

# ============================================================================
# Int Vector grids + Float32 data — the value-matched (Tg = Float32) arms
# ============================================================================
#
# `float(Int) === Float64`, so the Int-axis + Float64-data pins above never
# enter the eltype-mismatch arms (axes stay raw). Float32 data forces
# `Tg = Float32` and every Int axis must convert: PreCompute and the linear
# resolve go through POOLED wraps (zero heap), while the OnTheFly global
# collapse bridge still heap-converts mismatched axes (`Tg.(x)` per call) —
# pinned `@test_broken` on ≥ 1.12 until the bridge acquires from the pool.
# (On LTS the allocation floor exceeds the defect size, so no bound is pinned.)

@testitem "Int Vector grids + Float32 data (Tg = Float32 arms)" setup = [AllocConstants] begin
    function _alloc_linear_int_f32_2d()
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [0, 1, 2, 3, 4, 5]
        data = Float32[1.0f0 * a + 2.0f0 * b for a in x, b in y]
        q = (3.4f0, 2.6f0)
        for _ in 1:3
            linear_interp((x, y), data, q)
        end
        @allocated linear_interp((x, y), data, q)
    end
    function _alloc_cubic_int_pre_f32_2d()
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [0, 1, 2, 3, 4, 5]
        data = Float32[sin(1.0f0 * a) + cos(1.0f0 * b) for a in x, b in y]
        q = (3.4f0, 2.6f0)
        for _ in 1:3
            cubic_interp((x, y), data, q; coeffs = PreCompute())
        end
        @allocated cubic_interp((x, y), data, q; coeffs = PreCompute())
    end
    function _alloc_cubic_int_otf_f32_2d()
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [0, 1, 2, 3, 4, 5]
        data = Float32[sin(1.0f0 * a) + cos(1.0f0 * b) for a in x, b in y]
        q = (3.4f0, 2.6f0)
        for _ in 1:3
            cubic_interp((x, y), data, q)
        end
        @allocated cubic_interp((x, y), data, q)
    end
    function _alloc_hetero_mixed_int_f32_2d()
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [0, 1, 2, 3, 4, 5]
        data = Float32[sin(1.0f0 * a) + 2.0f0 * b for a in x, b in y]
        q = (3.4f0, 2.6f0)
        m = (CubicInterp(), LinearInterp())
        for _ in 1:3
            interp((x, y), data, q; method = m)
        end
        @allocated interp((x, y), data, q; method = m)
    end

    # value-match sanity: Int axes + Float32 data → Float32 out on every family
    @testset "results are Float32" begin
        x = [0, 1, 2, 3, 4]
        y = [0, 1, 2, 3]
        data = Float32[1.0f0 * a + 2.0f0 * b for a in x, b in y]
        q = (1.4f0, 0.6f0)
        @test linear_interp((x, y), data, q) isa Float32
        @test cubic_interp((x, y), data, q) isa Float32
        @test cubic_interp((x, y), data, q; coeffs = PreCompute()) isa Float32
        @test interp((x, y), data, q; method = (CubicInterp(), LinearInterp())) isa Float32
    end

    # Homogeneous arms are zero-alloc: the pooled PreCompute/linear wraps and — now
    # that the inner 1D cubic value-matches raw Int axes (data-aware cache) — the
    # homogeneous OnTheFly cubic collapse.
    @testset "warm zero-alloc (homogeneous raw-grid arms)" begin
        @test _alloc_linear_int_f32_2d() <= ND_ALLOC_THRESHOLD
        @test _alloc_cubic_int_pre_f32_2d() <= ND_ALLOC_THRESHOLD
        @test _alloc_cubic_int_otf_f32_2d() <= ND_ALLOC_THRESHOLD
    end

    # The mixed-method hetero fallback still eagerly converts the grids via
    # `_nd_promote_grids` (`_convert_grids_typed` → per-axis `Tg.(x)`, ~176 B on
    # an Int Vector axis) BEFORE the OnTheFly collapse — a separate convert arm
    # from the (deleted) `_bridge_axes_raw`. Making that dispatch pass raw grids
    # (type-only promotion) is a follow-up beyond the cubic-cache scope.
    @testset "mixed-method hetero fallback (RED until type-only promotion)" begin
        if ND_ALLOC_THRESHOLD == 0
            @test_broken _alloc_hetero_mixed_int_f32_2d() <= ND_ALLOC_THRESHOLD
        end
    end
end

# ============================================================================
# Quadratic ND one-shot — raw Int grid, default PreCompute path
# ============================================================================
#
# Quad scalar `AutoCoeffs` routes through the shared `_resolve_coeffs_nd_oneshot`
# policy (scalar → OnTheFly, batch → PreCompute), same as cubic. Both scalar
# strategies stay zero-alloc on raw Int Vector grids: OnTheFly via pooled collapse,
# explicit PreCompute via pooled value-matched axis conversion. Batch keeps eager-convert.

@testitem "Quadratic ND one-shot raw-grid (no eager convert)" setup = [AllocConstants] begin
    using ForwardDiff

    function _alloc_quad_nd_int_2d(coeffs)
        x = [0, 1, 2, 3, 4, 5, 6]
        y = [0, 1, 2, 3, 4]
        data = [sin(1.0 * a) + cos(1.0 * b) for a in x, b in y]
        q = (3.4, 2.6)
        for _ in 1:3
            quadratic_interp((x, y), data, q; coeffs)
        end
        @allocated quadratic_interp((x, y), data, q; coeffs)
    end
    function _alloc_quad_nd_int_3d(coeffs)
        x = [0, 1, 2, 3, 4]
        y = [0, 1, 2, 3]
        z = [0, 1, 2, 3, 4, 5]
        data = [a + 0.5b + 0.25c for a in x, b in y, c in z]
        q = (2.4, 1.6, 3.8)
        for _ in 1:3
            quadratic_interp((x, y, z), data, q; coeffs)
        end
        @allocated quadratic_interp((x, y, z), data, q; coeffs)
    end

    @testset "warm zero-alloc scalar one-shot on Int Vector grids ($label)" for (label, coeffs) in
        (("default → OnTheFly", AutoCoeffs()), ("explicit PreCompute", PreCompute()))
        @test _alloc_quad_nd_int_2d(coeffs) <= ND_ALLOC_THRESHOLD
        @test _alloc_quad_nd_int_3d(coeffs) <= ND_ALLOC_THRESHOLD
    end

    # ---- bit-identical: raw Int grid === Float64 grid (both scalar strategies) ----
    @testset "Int Vector grid === Float64 grid ($label)" for (label, coeffs) in
        (("default → OnTheFly", AutoCoeffs()), ("explicit PreCompute", PreCompute()))
        x = [0, 1, 2, 3, 4, 5, 6]
        y = [0, 1, 2, 3, 4]
        data = [sin(1.0 * a) + cos(1.0 * b) for a in x, b in y]
        xf = Float64.(x)
        yf = Float64.(y)
        for q in [(3.4, 2.6), (0.3, 0.7), (5.9, 3.1)]
            @test quadratic_interp((x, y), data, q; coeffs) ===
                quadratic_interp((xf, yf), data, q; coeffs)
        end
    end

    # ---- persistent parity: bit-exact is opt-in via coeffs=PreCompute();
    #      the default (OnTheFly) matches to FP-reordering noise ----
    @testset "one-shot vs persistent QuadraticInterpolantND" begin
        x = [0, 1, 2, 3, 4, 5, 6]
        y = [0, 1, 2, 3, 4]
        data = [sin(1.0 * a) + cos(1.0 * b) for a in x, b in y]
        itp = quadratic_interp((x, y), data)
        for q in [(3.4, 2.6), (0.3, 0.7), (5.9, 3.1)]
            @test quadratic_interp((x, y), data, q; coeffs = PreCompute()) === itp(q)
            @test quadratic_interp((x, y), data, q) ≈ itp(q) rtol = 1.0e-12
        end
    end

    # ---- type stability (::Tr) ----
    @testset "type-stable (::Tr) on Int grid" begin
        x = [0, 1, 2, 3, 4, 5, 6]
        y = [0, 1, 2, 3, 4]
        data = [sin(1.0 * a) + cos(1.0 * b) for a in x, b in y]
        q = (3.4, 2.6)
        @test (@inferred quadratic_interp((x, y), data, q)) isa Float64
    end

    # ---- ForwardDiff through the ND query on an Int Vector grid ----
    @testset "ForwardDiff through ND query on Int Vector grid" begin
        x = [0, 1, 2, 3, 4, 5, 6]
        y = [0, 1, 2, 3, 4]
        data = [1.0 * a^2 + 2.0 * b for a in x, b in y]
        p = [3.4, 2.6]
        g = ForwardDiff.gradient(pp -> quadratic_interp((x, y), data, (pp[1], pp[2])), p)
        gf = ForwardDiff.gradient(pp -> quadratic_interp((Float64.(x), Float64.(y)), data, (pp[1], pp[2])), p)
        @test g ≈ gf atol = 1.0e-10
    end
end

# ============================================================================
# Hermite ND one-shot — raw Int grid (user-supplied nodal partials)
# ============================================================================
#
# Hermite ND one-shot packs the user partials into a pool buffer and evaluates
# via the shared `_compute_all_local_params` (now floats the cell width) +
# `_eval_nd_cell`. Passing raw grids (no eager `Tg.(x)` copy) makes the warm
# scalar one-shot on an Int Vector grid zero-alloc. Batch keeps pool packing.

@testitem "Hermite ND one-shot raw-grid (no eager convert)" setup = [AllocConstants] begin
    using FastInterpolations: HermitePartials
    using ForwardDiff

    _hp(x, y) = HermitePartials(
        (1, 0) => [cos(1.0 * a) * cos(1.0 * b) for a in x, b in y],
        (0, 1) => [-sin(1.0 * a) * sin(1.0 * b) for a in x, b in y],
        (1, 1) => [-cos(1.0 * a) * sin(1.0 * b) for a in x, b in y],
    )
    _hdata(x, y) = [sin(1.0 * a) * cos(1.0 * b) for a in x, b in y]

    function _alloc_hermite_nd_int_2d()
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4]
        data = _hdata(x, y)
        p = _hp(x, y)
        q = (3.4, 2.6)
        for _ in 1:3
            hermite_interp((x, y), data, p, q)
        end
        @allocated hermite_interp((x, y), data, p, q)
    end

    @testset "warm zero-alloc scalar one-shot on Int Vector grid" begin
        @test _alloc_hermite_nd_int_2d() <= ND_ALLOC_THRESHOLD
    end

    # ---- bit-identical: raw Int grid === Float64 grid ----
    @testset "Int Vector grid === Float64 grid" begin
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4]
        data = _hdata(x, y)
        p = _hp(x, y)
        xf = Float64.(x)
        yf = Float64.(y)
        for q in [(3.4, 2.6), (0.3, 0.7), (4.9, 3.1)]
            @test hermite_interp((x, y), data, p, q) === hermite_interp((xf, yf), data, p, q)
        end
    end

    # ---- one-shot === persistent interpolant ----
    @testset "one-shot === persistent hermite interpolant" begin
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4]
        data = _hdata(x, y)
        p = _hp(x, y)
        itp = hermite_interp((x, y), data, p)
        for q in [(3.4, 2.6), (0.3, 0.7), (4.9, 3.1)]
            @test hermite_interp((x, y), data, p, q) === itp(q)
        end
    end

    # ---- type stability + ForwardDiff through the query ----
    @testset "type-stable + ForwardDiff on Int grid" begin
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4]
        data = _hdata(x, y)
        p = _hp(x, y)
        q = (3.4, 2.6)
        @test (@inferred hermite_interp((x, y), data, p, q)) isa Float64
        g = ForwardDiff.gradient(pp -> hermite_interp((x, y), data, p, (pp[1], pp[2])), [3.4, 2.6])
        gf = ForwardDiff.gradient(pp -> hermite_interp((Float64.(x), Float64.(y)), data, p, (pp[1], pp[2])), [3.4, 2.6])
        @test g ≈ gf atol = 1.0e-10
    end
end

# ============================================================================
# Heterogeneous / mixed-eltype grids on the build methods.
# ============================================================================
#
# Raw grids expose a homogeneity assumption: `_eval_nd_cell` couples `hs`/`inv_hs`
# as `NTuple{N, Tg}`, so mixed grids (Int×Float64, Float32×Float64, Range×Vector)
# must promote spacings to one `Tg` (via `_compute_all_local_params`).

@testitem "ND build-method one-shot — heterogeneous/mixed grids" setup = [AllocConstants] begin
    using FastInterpolations: HermitePartials

    fq(a, b) = sin(0.3 * a) + cos(0.2 * b)

    quad_mixes = [
        ("Int × Float64", [0, 1, 2, 3, 4, 5], Float64.(0:4)),
        ("Float32 × Float64", Float32.(0:5), Float64.(0:4)),
        ("Int × Float32", [0, 1, 2, 3, 4, 5], Float32.(0:4)),
        ("IntRange × IntVec", 0:5, [0, 1, 2, 3, 4]),
        ("Float64Range × IntVec", 0.0:5.0, [0, 1, 2, 3, 4]),
    ]
    for (name, gx, gy) in quad_mixes
        @testset "quadratic $name === all-Float64" begin
            data = [fq(a, b) for a in collect(gx), b in collect(gy)]
            ref = quadratic_interp((Float64.(collect(gx)), Float64.(collect(gy))), data, (3.4, 2.6))
            @test quadratic_interp((gx, gy), data, (3.4, 2.6)) ≈ ref rtol = 1.0e-6
        end
    end

    # Cubic: both the OnTheFly default and the (now-raw) PreCompute strategy.
    for (name, gx, gy) in quad_mixes
        @testset "cubic $name === all-Float64" begin
            data = [fq(a, b) for a in collect(gx), b in collect(gy)]
            gxf = Float64.(collect(gx))
            gyf = Float64.(collect(gy))
            @test cubic_interp((gx, gy), data, (3.4, 2.6)) ≈ cubic_interp((gxf, gyf), data, (3.4, 2.6)) rtol = 1.0e-6
            @test cubic_interp((gx, gy), data, (3.4, 2.6); coeffs = PreCompute()) ≈
                cubic_interp((gxf, gyf), data, (3.4, 2.6); coeffs = PreCompute()) rtol = 1.0e-6
        end
    end

    hp(x, y) = HermitePartials(
        (1, 0) => [cos(1.0 * a) * cos(1.0 * b) for a in x, b in y],
        (0, 1) => [-sin(1.0 * a) * sin(1.0 * b) for a in x, b in y],
        (1, 1) => [-cos(1.0 * a) * sin(1.0 * b) for a in x, b in y],
    )
    hdata(x, y) = [sin(1.0 * a) * cos(1.0 * b) for a in x, b in y]
    herm_mixes = [
        ("Float32 × Float64", Float32.(0:5), Float64.(0:4)),
        ("Int × Float32", [0, 1, 2, 3, 4, 5], Float32.(0:4)),
        ("IntRange × IntVec", 0:5, [0, 1, 2, 3, 4]),
    ]
    for (name, gx, gy) in herm_mixes
        @testset "hermite $name === all-Float64" begin
            xc = collect(gx)
            yc = collect(gy)
            data = hdata(xc, yc)
            p = hp(xc, yc)
            ref = hermite_interp((Float64.(xc), Float64.(yc)), data, p, (3.4, 2.6))
            @test hermite_interp((gx, gy), data, p, (3.4, 2.6)) ≈ ref rtol = 1.0e-6
        end
    end
end
