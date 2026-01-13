# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    CUBIC UNIFIED INTERPOLANT TESTS                        ║
# ║     Tests for CubicMultiInterpolantUnified - adaptive layout type         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ALLOC_THRESHOLD is defined in runtests.jl

using Test
using FastInterpolations
using FastInterpolations: TransposeSnapshot, CubicMultiInterpolantUnified
using FastInterpolations: _should_wrap, _ensure_point_layout!

# ============================================================================
# Phase 1: Type Construction Tests
# ============================================================================

@testset "CubicMultiInterpolantUnified" begin

@testset "Type Construction" begin
    @testset "TransposeSnapshot" begin
        # Empty snapshot creation
        snap = TransposeSnapshot{Float64}()
        @test snap.y_point === nothing
        @test snap.z_point === nothing

        # Snapshot with matrices
        y_point = rand(3, 10)
        z_point = rand(3, 10)
        snap2 = TransposeSnapshot{Float64}(y_point, z_point)
        @test snap2.y_point === y_point
        @test snap2.z_point === z_point
    end

    @testset "CubicMultiInterpolantUnified type parameters" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])

        # Check type hierarchy
        @test mitp isa AbstractMultiInterpolant{Float64}
        @test mitp isa CubicMultiInterpolantUnified

        # Check fields exist and have correct types
        @test mitp.y isa Matrix{Float64}
        @test mitp.z isa Matrix{Float64}
        @test size(mitp.y) == (11, 2)  # (n_points × n_series)
        @test size(mitp.z) == (11, 2)
    end

    @testset "Helper functions" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        # Regular BC
        mitp = cubic_interp_unified(x, [y1, y2]; extrap=:none)
        @test _should_wrap(mitp) == false

        # Wrap extrap
        mitp_wrap = cubic_interp_unified(x, [y1, y2]; extrap=:wrap)
        @test _should_wrap(mitp_wrap) == true
    end
end

# ============================================================================
# Phase 2: Vector{Vector} Constructor Tests
# ============================================================================

@testset "Vector{Vector} Constructor" begin
    @testset "Basic construction with NaturalBC" begin
        x = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        y3 = exp.(-x)

        mitp = cubic_interp_unified(x, [y1, y2, y3])

        @test size(mitp.y, 2) == 3  # 3 series
        @test size(mitp.y, 1) == 51  # 51 points
    end

    @testset "Type promotion" begin
        x = collect(1:10)  # Int
        y1 = collect(1:10)  # Int
        y2 = collect(11:20)  # Int

        mitp = cubic_interp_unified(x, [y1, y2])
        @test eltype(mitp.y) == Float64
    end

    @testset "Validation errors" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = rand(11)
        y_wrong = rand(10)  # Wrong length

        @test_throws AssertionError cubic_interp_unified(x, Vector{Float64}[])  # Empty
        @test_throws DimensionMismatch cubic_interp_unified(x, [y1, y_wrong])  # Length mismatch
    end

    @testset "Periodic BC endpoint check" begin
        x = collect(range(0.0, 1.0, 11))
        y_periodic = sin.(2π .* x)  # y[1] == y[end]
        y_non_periodic = copy(y_periodic)
        y_non_periodic[end] = y_non_periodic[end] + 0.5  # Break periodicity

        # Periodic with matching endpoints should work
        mitp = cubic_interp_unified(x, [y_periodic]; bc=PeriodicBC())
        @test _should_wrap(mitp) == true

        # Periodic with non-matching endpoints should fail
        @test_throws ArgumentError cubic_interp_unified(x, [y_non_periodic]; bc=PeriodicBC())
    end

    @testset "Coefficients match CubicMultiInterpolant" begin
        x = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        # Build both types
        mitp_unified = cubic_interp_unified(x, [y1, y2])
        mitp_ref = cubic_interp(x, [y1, y2])

        # Compare coefficients (z values)
        # Both now store in (n_points × n_series) layout
        @test mitp_unified.z ≈ mitp_ref.z rtol=1e-12
        @test mitp_unified.y ≈ mitp_ref.y rtol=1e-14
    end
end

# ============================================================================
# Phase 3: Matrix Constructor Tests
# ============================================================================

@testset "Matrix Constructor" begin
    @testset "Matrix input where columns are y-series" begin
        x = collect(range(0.0, 1.0, 51))
        Y = hcat(sin.(2π .* x), cos.(2π .* x), exp.(-x))  # 51×3 matrix

        mitp = cubic_interp_unified(x, Y)

        @test size(mitp.y) == (51, 3)
        @test mitp.y[:, 1] ≈ sin.(2π .* x)
        @test mitp.y[:, 2] ≈ cos.(2π .* x)
    end

    @testset "Type promotion for matrix elements" begin
        x = collect(1:10)
        Y = hcat(collect(1:10), collect(11:20))  # Int matrix

        mitp = cubic_interp_unified(x, Y)
        @test eltype(mitp.y) == Float64
    end

    @testset "Matrix vs Vector{Vector} equivalence" begin
        x = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        Y = hcat(y1, y2)

        mitp_vec = cubic_interp_unified(x, [y1, y2])
        mitp_mat = cubic_interp_unified(x, Y)

        @test mitp_vec.y ≈ mitp_mat.y
        @test mitp_vec.z ≈ mitp_mat.z
    end
end

# ============================================================================
# Phase 4: Lazy Point-Layout (Atomic Snapshot) Tests
# ============================================================================

@testset "Lazy Point-Layout" begin
    @testset "_ensure_point_layout! returns valid matrices" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])

        # Point-layout should be initially empty
        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point === nothing

        # Ensure creates valid matrices
        y_point, z_point = _ensure_point_layout!(mitp)
        @test size(y_point) == (2, 11)  # (n_series × n_points)
        @test size(z_point) == (2, 11)

        # Values should be transposed
        @test y_point[1, :] ≈ mitp.y[:, 1]  # Series 1
        @test y_point[2, :] ≈ mitp.y[:, 2]  # Series 2
    end

    @testset "precompute_transpose!" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])

        # Before precompute
        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point === nothing

        # After precompute
        precompute_transpose!(mitp)
        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point !== nothing
        @test snap.z_point !== nothing
    end

    @testset "Idempotent: calling twice returns same matrices" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])

        y_point1, z_point1 = _ensure_point_layout!(mitp)
        y_point2, z_point2 = _ensure_point_layout!(mitp)

        # Same matrices (by reference after first creation)
        @test y_point1 === y_point2
        @test z_point1 === z_point2
    end

    @testset "precompute_transpose=true at construction" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2]; precompute_transpose=true)

        # Should be populated immediately
        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point !== nothing
        @test snap.z_point !== nothing
    end

    @testset "Thread-safety (concurrent _ensure_point_layout!)" begin
        x = collect(range(0.0, 1.0, 51))
        ys = [sin.(k .* x) for k in 1:5]

        mitp = cubic_interp_unified(x, ys)

        # Spawn multiple tasks calling _ensure_point_layout! concurrently
        results = Vector{Any}(undef, 10)
        Threads.@threads for i in 1:10
            results[i] = _ensure_point_layout!(mitp)
        end

        # All should get valid results
        for (y_pt, z_pt) in results
            @test size(y_pt) == (5, 51)
            @test size(z_pt) == (5, 51)
        end

        # All should see same values
        y_ref, z_ref = results[1]
        for (y_pt, z_pt) in results
            @test y_pt ≈ y_ref
            @test z_pt ≈ z_ref
        end
    end
end

# ============================================================================
# Phase 5: Scalar Evaluation Tests
# ============================================================================

@testset "Scalar Evaluation" begin
    @testset "Basic scalar evaluation" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])

        # Out-of-place
        vals = mitp(0.5)
        @test length(vals) == 2
        @test vals[1] ≈ sin(π) atol=1e-6
        @test vals[2] ≈ cos(π) atol=1e-6
    end

    @testset "In-place evaluation" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])
        out = zeros(2)

        mitp(out, 0.5)
        @test out[1] ≈ sin(π) atol=1e-6
        @test out[2] ≈ cos(π) atol=1e-6
    end

    @testset "Results match CubicMultiInterpolantFused" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        y3 = exp.(-x)

        mitp_unified = cubic_interp_unified(x, [y1, y2, y3])
        mitp_fused = cubic_interp_fused(x, [y1, y2, y3])

        for xq in [0.1, 0.25, 0.5, 0.75, 0.9]
            vals_unified = mitp_unified(xq)
            vals_fused = mitp_fused(xq)
            @test vals_unified ≈ vals_fused rtol=1e-12
        end
    end

    @testset "Extrapolation modes" begin
        x = collect(range(0.0, 1.0, 51))
        y1 = x .^ 2

        # :none throws DomainError
        mitp_none = cubic_interp_unified(x, [y1]; extrap=:none)
        @test_throws DomainError mitp_none(-0.1)
        @test_throws DomainError mitp_none(1.1)

        # :constant returns boundary values
        mitp_const = cubic_interp_unified(x, [y1]; extrap=:constant)
        @test mitp_const(-0.1)[1] ≈ 0.0 atol=1e-12  # y[1] = 0^2 = 0
        @test mitp_const(1.1)[1] ≈ 1.0 atol=1e-12  # y[end] = 1^2 = 1

        # :extension uses polynomial extrapolation
        mitp_ext = cubic_interp_unified(x, [y1]; extrap=:extension)
        @test mitp_ext(-0.1)[1] isa Float64  # Should not throw
        @test mitp_ext(1.1)[1] isa Float64

        # :wrap wraps to domain
        mitp_wrap = cubic_interp_unified(x, [y1]; extrap=:wrap)
        @test mitp_wrap(1.1)[1] ≈ mitp_wrap(0.1)[1] rtol=1e-10
    end

    @testset "Scalar extrapolation with deriv=1" begin
        x = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x)

        # :constant returns 0 for derivatives outside domain
        mitp_const = cubic_interp_unified(x, [y1]; extrap=:constant)
        @test mitp_const(-0.1; deriv=1)[1] ≈ 0.0
        @test mitp_const(1.1; deriv=1)[1] ≈ 0.0

        # :extension uses polynomial extrapolation for derivatives
        mitp_ext = cubic_interp_unified(x, [y1]; extrap=:extension)
        @test mitp_ext(-0.1; deriv=1)[1] isa Float64
        @test mitp_ext(1.1; deriv=1)[1] isa Float64

        # :wrap wraps derivatives
        mitp_wrap = cubic_interp_unified(x, [y1]; extrap=:wrap)
        @test mitp_wrap(1.1; deriv=1)[1] ≈ mitp_wrap(0.1; deriv=1)[1] rtol=1e-6
    end

    @testset "Scalar extrapolation with deriv=2" begin
        x = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x)

        # :constant returns 0 for second derivatives outside domain
        mitp_const = cubic_interp_unified(x, [y1]; extrap=:constant)
        @test mitp_const(-0.1; deriv=2)[1] ≈ 0.0
        @test mitp_const(1.1; deriv=2)[1] ≈ 0.0

        # :extension uses polynomial extrapolation
        mitp_ext = cubic_interp_unified(x, [y1]; extrap=:extension)
        @test mitp_ext(-0.1; deriv=2)[1] isa Float64
        @test mitp_ext(1.1; deriv=2)[1] isa Float64

        # :wrap wraps second derivatives
        mitp_wrap = cubic_interp_unified(x, [y1]; extrap=:wrap)
        @test mitp_wrap(1.1; deriv=2)[1] ≈ mitp_wrap(0.1; deriv=2)[1] rtol=1e-4
    end
end

# ============================================================================
# Phase 6: Vector Evaluation Tests
# ============================================================================

@testset "Vector Evaluation" begin
    @testset "Basic vector evaluation" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])

        xq = [0.1, 0.5, 0.9]
        outputs = mitp(xq)

        @test length(outputs) == 2  # 2 series
        @test length(outputs[1]) == 3  # 3 query points
        @test outputs[1][2] ≈ sin(π) atol=1e-6
        @test outputs[2][2] ≈ cos(π) atol=1e-6
    end

    @testset "In-place vector evaluation" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])

        xq = [0.1, 0.5, 0.9]
        outputs = [zeros(3) for _ in 1:2]

        mitp(outputs, xq)
        @test outputs[1][2] ≈ sin(π) atol=1e-6
        @test outputs[2][2] ≈ cos(π) atol=1e-6
    end

    @testset "Results match CubicMultiInterpolant" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        y3 = exp.(-x)

        mitp_unified = cubic_interp_unified(x, [y1, y2, y3])
        mitp_ref = cubic_interp(x, [y1, y2, y3])

        xq = collect(range(0.1, 0.9, 10))
        outputs_unified = mitp_unified(xq)
        outputs_ref = mitp_ref(xq)

        for k in 1:3
            @test outputs_unified[k] ≈ outputs_ref[k] rtol=1e-12
        end
    end

    @testset "Pre-built anchor API" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])

        xq = [0.1, 0.5, 0.9]
        # Pre-build anchors
        aq_vec = FastInterpolations._anchor_query(x, xq)

        outputs = [zeros(3) for _ in 1:2]
        mitp(outputs, aq_vec)

        @test outputs[1][2] ≈ sin(π) atol=1e-6
        @test outputs[2][2] ≈ cos(π) atol=1e-6
    end

    @testset "Vector extrapolation modes" begin
        x = collect(range(0.0, 1.0, 51))
        y1 = x .^ 2
        y2 = x .^ 3

        # Queries including points outside domain
        xq_outside = [-0.1, 0.5, 1.1]

        # :none throws DomainError for vector queries with out-of-domain points
        mitp_none = cubic_interp_unified(x, [y1, y2]; extrap=:none)
        @test_throws DomainError mitp_none(xq_outside)

        # :constant returns boundary values for out-of-domain points
        mitp_const = cubic_interp_unified(x, [y1, y2]; extrap=:constant)
        outputs_const = mitp_const(xq_outside)
        @test outputs_const[1][1] ≈ 0.0 atol=1e-12  # y1(-0.1) → y1[1] = 0
        @test outputs_const[1][3] ≈ 1.0 atol=1e-12  # y1(1.1) → y1[end] = 1
        @test outputs_const[2][1] ≈ 0.0 atol=1e-12  # y2(-0.1) → y2[1] = 0
        @test outputs_const[2][3] ≈ 1.0 atol=1e-12  # y2(1.1) → y2[end] = 1

        # :extension uses polynomial extrapolation
        mitp_ext = cubic_interp_unified(x, [y1, y2]; extrap=:extension)
        outputs_ext = mitp_ext(xq_outside)
        @test outputs_ext[1][1] isa Float64  # Should not throw
        @test outputs_ext[1][3] isa Float64
        @test outputs_ext[2][1] isa Float64
        @test outputs_ext[2][3] isa Float64

        # :wrap wraps to domain
        mitp_wrap = cubic_interp_unified(x, [y1, y2]; extrap=:wrap)
        outputs_wrap = mitp_wrap(xq_outside)
        xq_wrapped = [0.1, 0.5, 0.9]
        outputs_ref = mitp_wrap(xq_wrapped)
        # 1.1 wraps to 0.1, -0.1 wraps to 0.9
        @test outputs_wrap[1][3] ≈ outputs_ref[1][1] rtol=1e-6  # 1.1 → 0.1
        @test outputs_wrap[1][1] ≈ outputs_ref[1][3] rtol=1e-6  # -0.1 → 0.9
    end

    @testset "Vector extrapolation with derivatives" begin
        x = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x)

        xq_outside = [-0.1, 0.5, 1.1]

        # :constant returns 0 for derivatives outside domain
        mitp_const = cubic_interp_unified(x, [y1]; extrap=:constant)
        outputs_d1 = mitp_const(xq_outside; deriv=1)
        @test outputs_d1[1][1] ≈ 0.0  # deriv at -0.1 is 0
        @test outputs_d1[1][3] ≈ 0.0  # deriv at 1.1 is 0
        @test outputs_d1[1][2] != 0.0  # deriv at 0.5 is not 0

        outputs_d2 = mitp_const(xq_outside; deriv=2)
        @test outputs_d2[1][1] ≈ 0.0  # second deriv at -0.1 is 0
        @test outputs_d2[1][3] ≈ 0.0  # second deriv at 1.1 is 0

        # :extension uses polynomial extrapolation for derivatives
        mitp_ext = cubic_interp_unified(x, [y1]; extrap=:extension)
        outputs_ext_d1 = mitp_ext(xq_outside; deriv=1)
        @test outputs_ext_d1[1][1] isa Float64
        @test outputs_ext_d1[1][3] isa Float64

        outputs_ext_d2 = mitp_ext(xq_outside; deriv=2)
        @test outputs_ext_d2[1][1] isa Float64
        @test outputs_ext_d2[1][3] isa Float64

        # :wrap wraps derivatives
        mitp_wrap = cubic_interp_unified(x, [y1]; extrap=:wrap)
        outputs_wrap_d1 = mitp_wrap(xq_outside; deriv=1)
        xq_ref = [0.9, 0.5, 0.1]  # wrapped points
        outputs_ref_d1 = mitp_wrap(xq_ref; deriv=1)
        @test outputs_wrap_d1[1][1] ≈ outputs_ref_d1[1][1] rtol=1e-6  # -0.1 → 0.9
        @test outputs_wrap_d1[1][3] ≈ outputs_ref_d1[1][3] rtol=1e-6  # 1.1 → 0.1
    end
end

# ============================================================================
# Phase 7: Derivatives Tests
# ============================================================================

@testset "Derivatives" begin
    @testset "Scalar deriv=1 matches numerical derivative" begin
        x = collect(range(0.0, 1.0, 201))
        y1 = sin.(2π .* x)

        mitp = cubic_interp_unified(x, [y1])

        xq = 0.3
        h = 1e-7
        numerical_deriv = (mitp(xq + h)[1] - mitp(xq - h)[1]) / (2h)
        analytical_deriv = mitp(xq; deriv=1)[1]

        @test analytical_deriv ≈ numerical_deriv rtol=1e-4
        @test analytical_deriv ≈ 2π * cos(2π * xq) rtol=1e-4
    end

    @testset "Scalar deriv=2 matches numerical second derivative" begin
        x = collect(range(0.0, 1.0, 201))
        y1 = sin.(2π .* x)

        mitp = cubic_interp_unified(x, [y1])

        xq = 0.3
        h = 1e-5
        numerical_deriv2 = (mitp(xq + h)[1] - 2 * mitp(xq)[1] + mitp(xq - h)[1]) / h^2
        analytical_deriv2 = mitp(xq; deriv=2)[1]

        @test analytical_deriv2 ≈ numerical_deriv2 rtol=1e-2
        @test analytical_deriv2 ≈ -4π^2 * sin(2π * xq) rtol=1e-2
    end

    @testset "Vector deriv=1" begin
        x = collect(range(0.0, 1.0, 201))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])

        # Use test points where derivatives are not near zero
        xq = [0.1, 0.3]
        derivs = mitp(xq; deriv=1)

        @test length(derivs) == 2
        @test derivs[1][1] ≈ 2π * cos(2π * 0.1) rtol=1e-3
        @test derivs[2][2] ≈ -2π * sin(2π * 0.3) rtol=1e-3
    end

    @testset "Vector deriv=2" begin
        x = collect(range(0.0, 1.0, 201))
        y1 = sin.(2π .* x)

        mitp = cubic_interp_unified(x, [y1])

        # Use test points where second derivatives are not near zero
        xq = [0.1, 0.25]
        derivs2 = mitp(xq; deriv=2)

        @test derivs2[1][1] ≈ -4π^2 * sin(2π * 0.1) rtol=0.05
        @test derivs2[1][2] ≈ -4π^2 * sin(2π * 0.25) rtol=0.05
    end

    @testset "Derivatives at domain boundaries" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = x .^ 2

        mitp = cubic_interp_unified(x, [y1]; extrap=:extension)

        # At x=0, d/dx(x^2) = 0
        @test mitp(0.0; deriv=1)[1] ≈ 0.0 atol=0.1
        # At x=1, d/dx(x^2) = 2
        @test mitp(1.0; deriv=1)[1] ≈ 2.0 atol=0.1
    end

    @testset "Derivatives with :constant extrap (returns 0)" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)

        mitp = cubic_interp_unified(x, [y1]; extrap=:constant)

        # Outside domain, derivative should be 0
        @test mitp(-0.1; deriv=1)[1] ≈ 0.0
        @test mitp(1.1; deriv=1)[1] ≈ 0.0
        @test mitp(-0.1; deriv=2)[1] ≈ 0.0
        @test mitp(1.1; deriv=2)[1] ≈ 0.0
    end
end

# ============================================================================
# Phase 8: Allocation Tests
# ============================================================================

@testset "Allocation Tests" begin
    @testset "Scalar in-place evaluation is zero-alloc after warmup" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2]; precompute_transpose=true)
        out = zeros(2)

        # Warmup
        mitp(out, 0.5)

        # Measure allocations
        allocs = @allocated mitp(out, 0.5)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Vector in-place evaluation is zero-alloc after warmup" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp_unified(x, [y1, y2])
        xq = collect(range(0.1, 0.9, 50))
        outputs = [zeros(50) for _ in 1:2]

        # Warmup
        mitp(outputs, xq)

        # Measure allocations
        allocs = @allocated mitp(outputs, xq)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Point-layout creation only allocates once" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)

        mitp = cubic_interp_unified(x, [y1])

        # First call creates layout
        _ensure_point_layout!(mitp)

        # Subsequent calls should not allocate
        allocs = @allocated _ensure_point_layout!(mitp)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Memory footprint verification (zero-cost abstraction)" begin
        x = collect(range(0.0, 1.0, 101))
        ys = [rand(101) for _ in 1:10]

        mitp = cubic_interp_unified(x, ys)

        # Before scalar query: point-layout not created
        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point === nothing

        size_before = Base.summarysize(mitp)

        # After scalar query: point-layout created
        _ = mitp(0.5)

        snap_after = @atomic :acquire mitp._point_snapshot
        @test snap_after.y_point !== nothing

        size_after = Base.summarysize(mitp)

        # Memory should roughly double (point-layout added)
        # Allow some tolerance for metadata overhead
        ratio = size_after / size_before
        @test 1.5 < ratio < 2.5
    end
end

# ============================================================================
# Phase 9: Export & Integration Tests
# ============================================================================

@testset "Export & Integration" begin
    @testset "Public API accessibility" begin
        # These should be exported
        @test isdefined(FastInterpolations, :CubicMultiInterpolantUnified)
        @test isdefined(FastInterpolations, :cubic_interp_unified)
        @test isdefined(FastInterpolations, :precompute_transpose!)
    end

    @testset "Conversion from CubicMultiInterpolantFused" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        fused = cubic_interp_fused(x, [y1, y2])
        unified = CubicMultiInterpolantUnified(fused)

        # Results should match
        @test unified(0.5) ≈ fused(0.5) rtol=1e-12
    end
end

end  # @testset "CubicMultiInterpolantUnified"
