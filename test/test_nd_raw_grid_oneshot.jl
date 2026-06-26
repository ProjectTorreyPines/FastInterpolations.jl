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
        x = [0, 1, 2, 3, 4]      # Int Vector axis — was `Float64.(x)` per call
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
