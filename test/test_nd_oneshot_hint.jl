using Test
using FastInterpolations

@testset "ND Oneshot Hint: State Updates & Zero-Alloc" begin

    # ========================================
    # Hint State Update — Scalar Queries
    # ========================================
    # Verify that hint[d][] is updated to the correct interval index
    # after each scalar oneshot call. Grid 0:1:10 → interval k has
    # x[k] <= xq < x[k+1], so query 3.5 → interval 4, query 4.5 → interval 5.

    @testset "Scalar hint update — linear" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))

        linear_interp((xs, ys), data, (3.5, 4.5); hint=hints)
        @test hints[1][] == 4
        @test hints[2][] == 5
    end

    @testset "Scalar hint update — constant" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))

        constant_interp((xs, ys), data, (3.5, 4.5); hint=hints)
        @test hints[1][] == 4
        @test hints[2][] == 5
    end

    @testset "Scalar hint update — cubic" begin
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))

        # Grid step = 0.5, query 3.5 → interval ceil(3.5/0.5) = 7? Let's compute:
        # grid[7] = 3.0, grid[8] = 3.5, query=3.5 → search should find ix=7 (3.0≤3.5<3.5 is false)
        # Actually, grid[8] = 3.5, so 3.5 at the boundary: search_direct gives floor((3.5-0.0)/0.5)+1 = 8
        # But clamped to n-1=20 if at endpoint. For 3.5 which is grid[8], it's x[8]≤3.5<x[9]? x[8]=3.5, x[9]=4.0 → yes, ix=8
        # For 4.5: floor(4.5/0.5)+1 = 10, grid[10]=4.5, grid[11]=5.0 → ix=10
        cubic_interp((xs, ys), data, (3.5, 4.5); hint=hints)
        @test hints[1][] == 8
        @test hints[2][] == 10
    end

    @testset "Scalar hint update — quadratic" begin
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))

        quadratic_interp((xs, ys), data, (3.5, 4.5); hint=hints)
        @test hints[1][] == 8
        @test hints[2][] == 10
    end

    @testset "Scalar hint update — vector grids (binary search path)" begin
        # Non-uniform vector grid to force _search_linear_binary! path
        xs = collect([0.0, 0.5, 1.5, 3.0, 5.0, 7.0, 8.5, 10.0])
        ys = collect([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        data = [Float64(x + y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))

        # query x=4.0: xs[4]=3.0 ≤ 4.0 < 5.0=xs[5] → ix=4
        # query y=6.5: ys[7]=6.0 ≤ 6.5 < 7.0=ys[8] → iy=7
        linear_interp((xs, ys), data, (4.0, 6.5); hint=hints)
        @test hints[1][] == 4
        @test hints[2][] == 7
    end

    # ========================================
    # Sequential Scalar Tracking
    # ========================================
    # Verify hint advances correctly when querying in sorted order.
    # The hint should progressively track forward positions.

    @testset "Sequential scalar calls track forward — linear" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))

        positions = [(1.5, 1.5), (3.5, 3.5), (5.5, 5.5), (7.5, 7.5), (9.5, 9.5)]
        expected_ix = [2, 4, 6, 8, 10]
        expected_iy = [2, 4, 6, 8, 10]

        for (k, (qx, qy)) in enumerate(positions)
            val = linear_interp((xs, ys), data, (qx, qy); hint=hints)
            @test val ≈ qx + qy atol=1e-12
            @test hints[1][] == expected_ix[k]
            @test hints[2][] == expected_iy[k]
        end
    end

    @testset "Sequential scalar calls track backward — linear" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        hints = (Ref(10), Ref(10))  # Start from end

        positions = [(9.5, 9.5), (7.5, 7.5), (5.5, 5.5), (3.5, 3.5), (1.5, 1.5)]
        expected_ix = [10, 8, 6, 4, 2]

        for (k, (qx, qy)) in enumerate(positions)
            val = linear_interp((xs, ys), data, (qx, qy); hint=hints)
            @test val ≈ qx + qy atol=1e-12
            @test hints[1][] == expected_ix[k]
            @test hints[2][] == expected_ix[k]
        end
    end

    # ========================================
    # Hint State Update — Batch (SoA)
    # ========================================
    # After SoA batch, hint should reflect the last query's interval.

    @testset "SoA batch hint updates to last query interval — linear" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        xqs = collect(0.5:1.0:9.5)  # [0.5, 1.5, ..., 9.5]
        yqs = collect(0.5:1.0:9.5)
        out = Vector{Float64}(undef, length(xqs))
        hints = (Ref(1), Ref(1))

        linear_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        # Last query: (9.5, 9.5) → interval 10
        @test hints[1][] == 10
        @test hints[2][] == 10
    end

    @testset "SoA batch hint updates to last query interval — constant" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        xqs = collect(0.5:1.0:9.5)
        yqs = collect(0.5:1.0:9.5)
        out = Vector{Float64}(undef, length(xqs))
        hints = (Ref(1), Ref(1))

        constant_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        @test hints[1][] == 10
        @test hints[2][] == 10
    end

    @testset "SoA batch hint updates to last query interval — cubic" begin
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]

        xqs = collect(0.5:1.0:9.5)
        yqs = collect(0.5:1.0:9.5)
        out = Vector{Float64}(undef, length(xqs))
        hints = (Ref(1), Ref(1))

        cubic_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        # Last query 9.5: floor(9.5/0.5)+1 = 20, clamped to 20 (n-1=20)
        @test hints[1][] == 20
        @test hints[2][] == 20
    end

    @testset "SoA batch hint updates to last query interval — quadratic" begin
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]

        xqs = collect(0.5:1.0:9.5)
        yqs = collect(0.5:1.0:9.5)
        out = Vector{Float64}(undef, length(xqs))
        hints = (Ref(1), Ref(1))

        quadratic_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        @test hints[1][] == 20
        @test hints[2][] == 20
    end

    # ========================================
    # Hint State Update — Batch (AoS)
    # ========================================

    @testset "AoS batch hint updates to last query interval — linear" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        points = [(0.5, 0.5), (3.5, 4.5), (7.5, 8.5)]
        out = Vector{Float64}(undef, length(points))
        hints = (Ref(1), Ref(1))

        linear_interp!(out, (xs, ys), data, points; hint=hints)
        # Last query: (7.5, 8.5) → ix=8, iy=9
        @test hints[1][] == 8
        @test hints[2][] == 9
    end

    # ========================================
    # 3D Hint Tests
    # ========================================

    @testset "3D hint correctness — linear" begin
        xs = 0.0:1.0:5.0
        ys = 0.0:1.0:5.0
        zs = 0.0:1.0:5.0
        data = [Float64(x + y + z) for x in xs, y in ys, z in zs]
        hints = (Ref(1), Ref(1), Ref(1))

        val_no = linear_interp((xs, ys, zs), data, (2.5, 3.5, 1.5))
        val_yes = linear_interp((xs, ys, zs), data, (2.5, 3.5, 1.5); hint=hints)
        @test val_no ≈ val_yes
        @test val_no ≈ 7.5 atol=1e-12
        @test hints[1][] == 3
        @test hints[2][] == 4
        @test hints[3][] == 2
    end

    @testset "3D hint correctness — cubic" begin
        xs = collect(range(0.0, 5.0, 11))
        ys = collect(range(0.0, 5.0, 11))
        zs = collect(range(0.0, 5.0, 11))
        data = [sin(x) * cos(y) * exp(-z/5) for x in xs, y in ys, z in zs]
        hints = (Ref(1), Ref(1), Ref(1))

        val_no = cubic_interp((xs, ys, zs), data, (2.5, 3.5, 1.5))
        val_yes = cubic_interp((xs, ys, zs), data, (2.5, 3.5, 1.5); hint=hints)
        @test val_no ≈ val_yes atol=1e-12
    end

    @testset "3D SoA batch with hints — linear" begin
        xs = 0.0:1.0:5.0
        ys = 0.0:1.0:5.0
        zs = 0.0:1.0:5.0
        data = [Float64(x + y + z) for x in xs, y in ys, z in zs]

        xqs = collect(0.5:1.0:4.5)
        yqs = collect(0.5:1.0:4.5)
        zqs = collect(0.5:1.0:4.5)
        n = length(xqs)

        out1 = Vector{Float64}(undef, n)
        linear_interp!(out1, (xs, ys, zs), data, (xqs, yqs, zqs))

        hints = (Ref(1), Ref(1), Ref(1))
        out2 = Vector{Float64}(undef, n)
        linear_interp!(out2, (xs, ys, zs), data, (xqs, yqs, zqs); hint=hints)

        @test out1 ≈ out2
        # Last query (4.5, 4.5, 4.5) → interval 5 on each axis
        @test hints[1][] == 5
        @test hints[2][] == 5
        @test hints[3][] == 5
    end

    # ========================================
    # Hint with Explicit Search Policy
    # ========================================

    @testset "Hint with explicit LinearBinarySearch policy — linear" begin
        xs = collect([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ys = collect([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        data = [Float64(x + y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))

        val = linear_interp((xs, ys), data, (5.5, 6.5);
            search=FastInterpolations.LinearBinarySearch(), hint=hints)
        @test val ≈ 12.0 atol=1e-12
        @test hints[1][] == 6
        @test hints[2][] == 7
    end

    # ========================================
    # Allocation Tests (function barriers)
    # ========================================
    # True function barriers: setup + warmup + @allocated all in ONE function.
    # This avoids @testset try/catch type-instability artifacts on @allocated.

    # --- Scalar with hint ---

    function _alloc_scalar_linear_hint()
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))
        linear_interp((xs, ys), data, (3.5, 4.5); hint=hints)
        linear_interp((xs, ys), data, (5.5, 6.5); hint=hints)
        @allocated linear_interp((xs, ys), data, (7.5, 8.5); hint=hints)
    end

    function _alloc_scalar_constant_hint()
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))
        constant_interp((xs, ys), data, (3.5, 4.5); hint=hints)
        constant_interp((xs, ys), data, (5.5, 6.5); hint=hints)
        @allocated constant_interp((xs, ys), data, (7.5, 8.5); hint=hints)
    end

    function _alloc_scalar_cubic_hint()
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))
        cubic_interp((xs, ys), data, (3.5, 4.5); hint=hints)
        cubic_interp((xs, ys), data, (5.5, 6.5); hint=hints)
        @allocated cubic_interp((xs, ys), data, (7.5, 8.5); hint=hints)
    end

    function _alloc_scalar_quadratic_hint()
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]
        hints = (Ref(1), Ref(1))
        quadratic_interp((xs, ys), data, (3.5, 4.5); hint=hints)
        quadratic_interp((xs, ys), data, (5.5, 6.5); hint=hints)
        @allocated quadratic_interp((xs, ys), data, (7.5, 8.5); hint=hints)
    end

    @testset "Zero-alloc: scalar with hint" begin
        @testset "linear" begin
            @test _alloc_scalar_linear_hint() == 0
        end
        @testset "constant" begin
            @test _alloc_scalar_constant_hint() == 0
        end
        @testset "cubic" begin
            @test _alloc_scalar_cubic_hint() == 0
        end
        @testset "quadratic" begin
            @test _alloc_scalar_quadratic_hint() == 0
        end
    end

    # --- SoA batch with hint (all 4 types) ---

    function _alloc_soa_linear_hint()
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        xqs = collect(0.5:0.5:9.5)
        yqs = collect(0.5:0.5:9.5)
        out = Vector{Float64}(undef, length(xqs))
        hints = (Ref(1), Ref(1))
        linear_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        linear_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        @allocated linear_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
    end

    function _alloc_soa_constant_hint()
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        xqs = collect(0.5:0.5:9.5)
        yqs = collect(0.5:0.5:9.5)
        out = Vector{Float64}(undef, length(xqs))
        hints = (Ref(1), Ref(1))
        constant_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        constant_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        @allocated constant_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
    end

    function _alloc_soa_cubic_hint()
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]
        xqs = collect(range(0.5, 9.5, 19))
        yqs = collect(range(0.5, 9.5, 19))
        out = Vector{Float64}(undef, length(xqs))
        hints = (Ref(1), Ref(1))
        cubic_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        cubic_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        @allocated cubic_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
    end

    function _alloc_soa_quadratic_hint()
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]
        xqs = collect(range(0.5, 9.5, 19))
        yqs = collect(range(0.5, 9.5, 19))
        out = Vector{Float64}(undef, length(xqs))
        hints = (Ref(1), Ref(1))
        quadratic_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        quadratic_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
        @allocated quadratic_interp!(out, (xs, ys), data, (xqs, yqs); hint=hints)
    end

    @testset "Zero-alloc: SoA batch with hint" begin
        @testset "linear" begin
            @test _alloc_soa_linear_hint() == 0
        end
        @testset "constant" begin
            @test _alloc_soa_constant_hint() == 0
        end
        @testset "cubic" begin
            @test _alloc_soa_cubic_hint() == 0
        end
        @testset "quadratic" begin
            @test _alloc_soa_quadratic_hint() == 0
        end
    end

    # --- AoS batch with hint ---

    function _alloc_aos_linear_hint()
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        points = [(0.5, 0.5), (3.5, 4.5), (7.5, 8.5)]
        out = Vector{Float64}(undef, length(points))
        hints = (Ref(1), Ref(1))
        linear_interp!(out, (xs, ys), data, points; hint=hints)
        linear_interp!(out, (xs, ys), data, points; hint=hints)
        @allocated linear_interp!(out, (xs, ys), data, points; hint=hints)
    end

    function _alloc_aos_cubic_hint()
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]
        points = [(1.5, 1.5), (3.5, 4.5), (7.5, 8.5)]
        out = Vector{Float64}(undef, length(points))
        hints = (Ref(1), Ref(1))
        cubic_interp!(out, (xs, ys), data, points; hint=hints)
        cubic_interp!(out, (xs, ys), data, points; hint=hints)
        @allocated cubic_interp!(out, (xs, ys), data, points; hint=hints)
    end

    @testset "Zero-alloc: AoS batch with hint" begin
        @testset "linear" begin
            @test _alloc_aos_linear_hint() == 0
        end
        @testset "cubic" begin
            @test _alloc_aos_cubic_hint() == 0
        end
    end
end
