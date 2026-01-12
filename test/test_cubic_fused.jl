# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      CUBIC FUSED INTERPOLANT TESTS                         ║
# ║         Tests for CubicMultiInterpolantFused with interleaved layout       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

using Test
using FastInterpolations

@testset "CubicMultiInterpolantFused" begin

    # ========================================
    # Phase 1: Type Definition Tests
    # ========================================
    @testset "Phase 1: Type Definition" begin

        @testset "Type hierarchy" begin
            # CubicMultiInterpolantFused should be subtype of AbstractMultiInterpolant
            @test CubicMultiInterpolantFused <: AbstractMultiInterpolant
        end

        @testset "Type can be constructed manually" begin
            T = Float64
            n_points = 5
            n_series = 3

            # Create minimal valid data
            x = collect(range(0.0, 1.0, n_points))
            spacing = FastInterpolations._create_spacing(x)
            y = Matrix{T}(undef, n_series, n_points)
            z = Matrix{T}(undef, n_series, n_points)

            # Fill with simple data
            for i in 1:n_series
                y[i, :] .= range(0.0, 1.0, n_points) .* i
                z[i, :] .= 0.0  # Natural BC
            end

            # BCPair for natural BC
            bc_config = BCPair(Deriv2(0.0), Deriv2(0.0))
            extrap = Val(:none)

            # Construct directly
            mitp = CubicMultiInterpolantFused(
                x, spacing, y, z, bc_config, extrap, n_series, n_points
            )

            @test mitp isa CubicMultiInterpolantFused
            @test mitp isa AbstractMultiInterpolant{Float64}
        end

        @testset "Field accessors return expected types" begin
            T = Float64
            n_points = 5
            n_series = 3

            x = collect(range(0.0, 1.0, n_points))
            spacing = FastInterpolations._create_spacing(x)
            y = ones(T, n_series, n_points)
            z = zeros(T, n_series, n_points)
            bc_config = BCPair(Deriv2(0.0), Deriv2(0.0))
            extrap = Val(:none)

            mitp = CubicMultiInterpolantFused(
                x, spacing, y, z, bc_config, extrap, n_series, n_points
            )

            # Test field access
            @test mitp.x === x
            @test mitp.spacing === spacing
            @test mitp.y === y
            @test mitp.z === z
            @test mitp.bc_config === bc_config
            @test mitp.extrap === extrap
            @test mitp.n_series == n_series
            @test mitp.n_points == n_points
        end

        @testset "Type parameters are correctly inferred" begin
            # Float64 with Vector grid
            x64 = collect(range(0.0, 1.0, 5))
            spacing64 = FastInterpolations._create_spacing(x64)
            y64 = ones(Float64, 3, 5)
            z64 = zeros(Float64, 3, 5)
            bc64 = BCPair(Deriv2(0.0), Deriv2(0.0))

            mitp64 = CubicMultiInterpolantFused(
                x64, spacing64, y64, z64, bc64, Val(:none), 3, 5
            )
            @test mitp64 isa CubicMultiInterpolantFused{Float64}

            # Float32 with Range grid
            x32 = range(0.0f0, 1.0f0, 5)
            spacing32 = FastInterpolations._create_spacing(x32)
            y32 = ones(Float32, 2, 5)
            z32 = zeros(Float32, 2, 5)
            bc32 = BCPair(Deriv2(0.0f0), Deriv2(0.0f0))

            mitp32 = CubicMultiInterpolantFused(
                x32, spacing32, y32, z32, bc32, Val(:constant), 2, 5
            )
            @test mitp32 isa CubicMultiInterpolantFused{Float32}
        end

    end  # Phase 1 testset

    # ========================================
    # Phase 2: Vector{Vector} Constructor Tests
    # ========================================
    @testset "Phase 2: Constructor" begin

        @testset "Basic construction with Vector{Vector}" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            y3 = exp.(-x)

            mitp = cubic_interp_fused(x, [y1, y2, y3])

            @test mitp isa CubicMultiInterpolantFused{Float64}
            @test mitp.n_series == 3
            @test mitp.n_points == 11
        end

        @testset "Coefficients match single CubicInterpolant" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Build fused interpolant
            mitp = cubic_interp_fused(x, [y1, y2])

            # Build individual interpolants
            itp1 = cubic_interp(x, y1)
            itp2 = cubic_interp(x, y2)

            # Check z coefficients match (within floating point tolerance)
            @test mitp.z[1, :] ≈ itp1.z atol=1e-14
            @test mitp.z[2, :] ≈ itp2.z atol=1e-14

            # Check y values match
            @test mitp.y[1, :] ≈ y1 atol=1e-14
            @test mitp.y[2, :] ≈ y2 atol=1e-14
        end

        @testset "Works with AbstractRange grid" begin
            x = range(0.0, 1.0, 11)  # Range, not collected
            y1 = sin.(2π .* collect(x))
            y2 = cos.(2π .* collect(x))

            mitp = cubic_interp_fused(x, [y1, y2])

            @test mitp isa CubicMultiInterpolantFused
            # Verify Range type is preserved for O(1) lookup
            @test mitp.x isa AbstractRange
        end

        @testset "Validation: mismatched y lengths throw DimensionMismatch" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x[1:5])  # Wrong length!

            @test_throws DimensionMismatch cubic_interp_fused(x, [y1, y2])
        end

        @testset "Validation: empty ys throws ArgumentError" begin
            x = collect(range(0.0, 1.0, 11))
            ys = Vector{Float64}[]

            @test_throws ArgumentError cubic_interp_fused(x, ys)
        end

        @testset "Works with bc keyword" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Test ClampedBC
            mitp_clamped = cubic_interp_fused(x, [y1, y2]; bc=ClampedBC())
            @test mitp_clamped isa CubicMultiInterpolantFused

            # Test explicit NaturalBC
            mitp_natural = cubic_interp_fused(x, [y1, y2]; bc=NaturalBC())
            @test mitp_natural isa CubicMultiInterpolantFused
        end

        @testset "Works with extrap keyword" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            for extrap in [:none, :constant, :extension]
                mitp = cubic_interp_fused(x, [y1, y2]; extrap=extrap)
                @test mitp.extrap === Val(extrap)
            end
        end

    end  # Phase 2 testset

    # ========================================
    # Phase 3: Scalar Evaluation Kernel (Value)
    # ========================================
    @testset "Phase 3: Scalar Evaluation" begin

        @testset "Out-of-place scalar: mitp(xq) returns Vector" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            y3 = exp.(-x)

            mitp = cubic_interp_fused(x, [y1, y2, y3])

            vals = mitp(0.5)

            @test vals isa Vector{Float64}
            @test length(vals) == 3
        end

        @testset "In-place scalar: mitp(out, xq) fills pre-allocated" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])

            out = zeros(2)
            result = mitp(out, 0.5)

            @test result === out  # Returns same array
            @test out[1] != 0.0   # Was actually filled
            @test out[2] != 0.0
        end

        @testset "Results match CubicMultiInterpolant" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            y3 = exp.(-x)

            mitp_fused = cubic_interp_fused(x, [y1, y2, y3])
            mitp_comp = cubic_interp(x, [y1, y2, y3])

            # Test at multiple points
            for xq in [0.1, 0.25, 0.5, 0.75, 0.9]
                vals_fused = mitp_fused(xq)
                vals_comp = mitp_comp(xq)

                @test vals_fused ≈ vals_comp atol=1e-14
            end
        end

        @testset "Evaluation at grid knots is exact" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])

            # At grid points, interpolation should match y exactly
            for (i, xi) in enumerate(x)
                vals = mitp(xi)
                @test vals[1] ≈ y1[i] atol=1e-14
                @test vals[2] ≈ y2[i] atol=1e-14
            end
        end

        @testset "DimensionMismatch for wrong output length" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])

            out_wrong = zeros(5)  # Wrong length!
            @test_throws DimensionMismatch mitp(out_wrong, 0.5)
        end

        @testset "Works at grid boundaries" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])

            # At boundaries
            vals_start = mitp(0.0)
            vals_end = mitp(1.0)

            @test vals_start ≈ [y1[1], y2[1]] atol=1e-14
            @test vals_end ≈ [y1[end], y2[end]] atol=1e-14
        end

    end  # Phase 3 testset

    # ========================================
    # Phase 4: Extrapolation Handling
    # ========================================
    @testset "Phase 4: Extrapolation" begin

        @testset ":none mode throws DomainError" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2]; extrap=:none)

            # Below domain
            @test_throws DomainError mitp(-0.1)
            # Above domain
            @test_throws DomainError mitp(1.1)
        end

        @testset ":constant mode returns boundary values" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2]; extrap=:constant)

            # Below domain: should return first values
            vals_below = mitp(-0.5)
            @test vals_below ≈ [y1[1], y2[1]] atol=1e-14

            # Above domain: should return last values
            vals_above = mitp(1.5)
            @test vals_above ≈ [y1[end], y2[end]] atol=1e-14
        end

        @testset ":constant mode matches CubicMultiInterpolant" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp_fused = cubic_interp_fused(x, [y1, y2]; extrap=:constant)
            mitp_comp = cubic_interp(x, [y1, y2]; extrap=:constant)

            # Test at various out-of-domain points
            for xq in [-0.5, -0.1, 1.1, 1.5]
                vals_fused = mitp_fused(xq)
                vals_comp = mitp_comp(xq)
                @test vals_fused ≈ vals_comp atol=1e-14
            end
        end

        @testset ":extension mode extrapolates smoothly" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp_fused = cubic_interp_fused(x, [y1, y2]; extrap=:extension)
            mitp_comp = cubic_interp(x, [y1, y2]; extrap=:extension)

            # Test at various out-of-domain points
            for xq in [-0.1, -0.05, 1.05, 1.1]
                vals_fused = mitp_fused(xq)
                vals_comp = mitp_comp(xq)
                @test vals_fused ≈ vals_comp atol=1e-14
            end

            # Extension should produce non-boundary values
            vals_below = mitp_fused(-0.1)
            @test vals_below[1] != y1[1]  # Not just clamped
        end

        @testset ":wrap mode wraps to domain" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp_fused = cubic_interp_fused(x, [y1, y2]; extrap=:wrap)
            mitp_comp = cubic_interp(x, [y1, y2]; extrap=:wrap)

            # Test periodic wrapping
            for xq in [-0.5, 1.5, 2.3, -1.7]
                vals_fused = mitp_fused(xq)
                vals_comp = mitp_comp(xq)
                @test vals_fused ≈ vals_comp atol=1e-14
            end

            # x + period should give same result
            @test mitp_fused(0.3) ≈ mitp_fused(1.3) atol=1e-14
            @test mitp_fused(0.7) ≈ mitp_fused(-0.3) atol=1e-14
        end

        @testset "All extrap modes work with in-place API" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            out = zeros(2)

            # :constant
            mitp_const = cubic_interp_fused(x, [y1, y2]; extrap=:constant)
            result = mitp_const(out, -0.5)
            @test result === out
            @test out ≈ [y1[1], y2[1]] atol=1e-14

            # :extension
            mitp_ext = cubic_interp_fused(x, [y1, y2]; extrap=:extension)
            mitp_ext(out, 1.1)
            @test out[1] != 0.0  # Was filled

            # :wrap
            mitp_wrap = cubic_interp_fused(x, [y1, y2]; extrap=:wrap)
            mitp_wrap(out, 1.3)
            @test out ≈ mitp_wrap(0.3) atol=1e-14
        end

    end  # Phase 4 testset

    # ========================================
    # Phase 5: Boundary Condition Support
    # ========================================
    @testset "Phase 5: Boundary Conditions" begin

        @testset "ClampedBC constructs successfully" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2]; bc=ClampedBC())
            @test mitp isa CubicMultiInterpolantFused
            @test mitp.n_series == 2
        end

        @testset "ClampedBC coefficients match single-series" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp_fused = cubic_interp_fused(x, [y1, y2]; bc=ClampedBC())
            itp1 = cubic_interp(x, y1; bc=ClampedBC())
            itp2 = cubic_interp(x, y2; bc=ClampedBC())

            # Check z coefficients match
            @test mitp_fused.z[1, :] ≈ itp1.z atol=1e-14
            @test mitp_fused.z[2, :] ≈ itp2.z atol=1e-14
        end

        @testset "ClampedBC evaluation matches CubicMultiInterpolant" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp_fused = cubic_interp_fused(x, [y1, y2]; bc=ClampedBC())
            mitp_comp = cubic_interp(x, [y1, y2]; bc=ClampedBC())

            for xq in [0.1, 0.25, 0.5, 0.75, 0.9]
                @test mitp_fused(xq) ≈ mitp_comp(xq) atol=1e-14
            end
        end

        @testset "PeriodicBC constructs successfully with periodic data" begin
            x = collect(range(0.0, 1.0, 21))
            # Create truly periodic data (y[1] == y[end])
            y1 = sin.(2π .* x)  # sin(0) == sin(2π) == 0
            y2 = cos.(2π .* x)  # cos(0) == cos(2π) == 1

            mitp = cubic_interp_fused(x, [y1, y2]; bc=PeriodicBC())
            @test mitp isa CubicMultiInterpolantFused
            @test mitp.n_series == 2
        end

        @testset "PeriodicBC forces :wrap extrapolation" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Even if we specify :none, PeriodicBC should force :wrap
            mitp = cubic_interp_fused(x, [y1, y2]; bc=PeriodicBC(), extrap=:none)
            @test mitp.extrap === Val(:wrap)

            # Should NOT throw for out-of-bounds query (wrap enabled)
            vals = mitp(1.5)
            @test vals ≈ mitp(0.5) atol=1e-12  # Period is 1.0
        end

        @testset "PeriodicBC throws for non-periodic y data" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = exp.(-x)  # NOT periodic: y[1]=1.0, y[end]=exp(-1)≈0.37
            y2 = sin.(2π .* x)  # periodic

            @test_throws ArgumentError cubic_interp_fused(x, [y1, y2]; bc=PeriodicBC())
        end

        @testset "PeriodicBC coefficients match single-series" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp_fused = cubic_interp_fused(x, [y1, y2]; bc=PeriodicBC())
            itp1 = cubic_interp(x, y1; bc=PeriodicBC())
            itp2 = cubic_interp(x, y2; bc=PeriodicBC())

            # Check z coefficients match
            @test mitp_fused.z[1, :] ≈ itp1.z atol=1e-14
            @test mitp_fused.z[2, :] ≈ itp2.z atol=1e-14
        end

        @testset "BCPair with custom Deriv1 values" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Custom first derivative values at boundaries
            bc = BCPair(Deriv1(0.0), Deriv1(0.0))
            mitp = cubic_interp_fused(x, [y1, y2]; bc=bc)

            @test mitp isa CubicMultiInterpolantFused
        end

        @testset "BCPair with custom Deriv2 values (same as NaturalBC)" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Deriv2(0) at both ends is equivalent to NaturalBC
            bc = BCPair(Deriv2(0.0), Deriv2(0.0))
            mitp_custom = cubic_interp_fused(x, [y1, y2]; bc=bc)
            mitp_natural = cubic_interp_fused(x, [y1, y2]; bc=NaturalBC())

            # Coefficients should match
            @test mitp_custom.z ≈ mitp_natural.z atol=1e-14
        end

        @testset "All BC types evaluate correctly at interior points" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            bcs = [NaturalBC(), ClampedBC(), PeriodicBC()]
            for bc in bcs
                mitp_fused = cubic_interp_fused(x, [y1, y2]; bc=bc)
                mitp_comp = cubic_interp(x, [y1, y2]; bc=bc)

                # Interior point evaluation should match
                for xq in [0.2, 0.4, 0.6, 0.8]
                    @test mitp_fused(xq) ≈ mitp_comp(xq) atol=1e-14
                end
            end
        end

    end  # Phase 5 testset

    # ========================================
    # Phase 6: Derivative Support
    # ========================================
    @testset "Phase 6: Derivatives" begin

        @testset "deriv=1 returns first derivatives" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])

            d1 = mitp(0.5; deriv=1)
            @test d1 isa Vector{Float64}
            @test length(d1) == 2

            # Analytical derivatives: d/dx sin(2πx) = 2π cos(2πx), d/dx cos(2πx) = -2π sin(2πx)
            # At x=0.5: cos(π)=-1, sin(π)=0
            # So d1 ≈ [-2π, 0]
            @test d1[1] ≈ -2π atol=1e-2  # Numerical vs analytical
            @test abs(d1[2]) < 1e-2  # Should be near zero
        end

        @testset "deriv=2 returns second derivatives" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])

            d2 = mitp(0.5; deriv=2)
            @test d2 isa Vector{Float64}
            @test length(d2) == 2

            # Analytical 2nd derivatives: d²/dx² sin(2πx) = -4π² sin(2πx), d²/dx² cos(2πx) = -4π² cos(2πx)
            # At x=0.5: sin(π)=0, cos(π)=-1
            # So d2 ≈ [0, 4π²]
            @test abs(d2[1]) < 1e-1  # Should be near zero
            @test d2[2] ≈ 4π^2 atol=1  # Numerical vs analytical
        end

        @testset "Derivatives match CubicMultiInterpolant" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            y3 = exp.(-x)

            mitp_fused = cubic_interp_fused(x, [y1, y2, y3])
            mitp_comp = cubic_interp(x, [y1, y2, y3])

            for xq in [0.1, 0.25, 0.5, 0.75, 0.9]
                # First derivative
                d1_fused = mitp_fused(xq; deriv=1)
                d1_comp = mitp_comp(xq; deriv=1)
                @test d1_fused ≈ d1_comp atol=1e-14

                # Second derivative
                d2_fused = mitp_fused(xq; deriv=2)
                d2_comp = mitp_comp(xq; deriv=2)
                @test d2_fused ≈ d2_comp atol=1e-14
            end
        end

        @testset "Derivatives work with in-place API" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])
            out = zeros(2)

            # deriv=1
            result = mitp(out, 0.5; deriv=1)
            @test result === out
            @test abs(out[1] + 2π) < 1e-2  # ≈ -2π

            # deriv=2
            mitp(out, 0.5; deriv=2)
            @test abs(out[2] - 4π^2) < 1  # ≈ 4π²
        end

        @testset "Derivatives match finite difference approximation" begin
            x = collect(range(0.0, 1.0, 101))
            y1 = x .^ 3 .- 2 .* x .^ 2 .+ x  # Cubic polynomial
            y2 = sin.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])

            # Test point
            xq = 0.4
            h = 1e-6

            # First derivative via finite difference
            v_plus = mitp(xq + h)
            v_minus = mitp(xq - h)
            fd1 = (v_plus .- v_minus) ./ (2h)
            d1 = mitp(xq; deriv=1)
            @test d1 ≈ fd1 atol=1e-5

            # Second derivative via finite difference
            v_center = mitp(xq)
            fd2 = (v_plus .- 2 .* v_center .+ v_minus) ./ h^2
            d2 = mitp(xq; deriv=2)
            @test d2 ≈ fd2 atol=1e-3  # Lower accuracy for 2nd deriv
        end

        @testset "Derivatives work with different BC types" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            for bc in [NaturalBC(), ClampedBC(), PeriodicBC()]
                mitp_fused = cubic_interp_fused(x, [y1, y2]; bc=bc)
                mitp_comp = cubic_interp(x, [y1, y2]; bc=bc)

                # Test at interior point
                d1_fused = mitp_fused(0.5; deriv=1)
                d1_comp = mitp_comp(0.5; deriv=1)
                @test d1_fused ≈ d1_comp atol=1e-14

                d2_fused = mitp_fused(0.5; deriv=2)
                d2_comp = mitp_comp(0.5; deriv=2)
                @test d2_fused ≈ d2_comp atol=1e-14
            end
        end

        @testset "Invalid deriv value throws error" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)

            mitp = cubic_interp_fused(x, [y1])

            # deriv=3 should throw
            @test_throws Exception mitp(0.5; deriv=3)
        end

    end  # Phase 6 testset

    # ========================================
    # Phase 7: Vector API (Multiple Queries)
    # ========================================
    @testset "Phase 7: Vector API" begin

        @testset "Out-of-place: mitp(xq_vec) returns Vector{Vector}" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            y3 = exp.(-x)

            mitp = cubic_interp_fused(x, [y1, y2, y3])
            xq_vec = [0.1, 0.3, 0.5, 0.7, 0.9]

            outputs = mitp(xq_vec)

            @test outputs isa Vector{Vector{Float64}}
            @test length(outputs) == 3  # n_series
            @test all(length(o) == 5 for o in outputs)  # n_query
        end

        @testset "Output layout is series-first" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])
            xq_vec = [0.2, 0.4, 0.6]

            outputs = mitp(xq_vec)

            # outputs[k][j] = series k at xq[j]
            for (j, xq) in enumerate(xq_vec)
                scalar_result = mitp(xq)
                @test outputs[1][j] ≈ scalar_result[1] atol=1e-14
                @test outputs[2][j] ≈ scalar_result[2] atol=1e-14
            end
        end

        @testset "In-place: mitp(outputs, xq_vec) fills pre-allocated" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])
            xq_vec = [0.1, 0.3, 0.5, 0.7, 0.9]

            outputs = [zeros(5) for _ in 1:2]
            result = mitp(outputs, xq_vec)

            @test result === outputs
            @test all(o -> any(v != 0.0 for v in o), outputs)  # Was filled
        end

        @testset "Results match CubicMultiInterpolant" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            y3 = exp.(-x)

            mitp_fused = cubic_interp_fused(x, [y1, y2, y3])
            mitp_comp = cubic_interp(x, [y1, y2, y3])

            xq_vec = [0.1, 0.25, 0.5, 0.75, 0.9]

            outputs_fused = mitp_fused(xq_vec)
            outputs_comp = mitp_comp(xq_vec)

            for k in 1:3
                @test outputs_fused[k] ≈ outputs_comp[k] atol=1e-14
            end
        end

        @testset "DimensionMismatch for wrong output lengths" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])
            xq_vec = [0.1, 0.3, 0.5]

            # Wrong number of series
            outputs_wrong_series = [zeros(3)]  # Only 1 series, need 2
            @test_throws DimensionMismatch mitp(outputs_wrong_series, xq_vec)

            # Wrong query length
            outputs_wrong_query = [zeros(5), zeros(5)]  # Wrong length
            @test_throws DimensionMismatch mitp(outputs_wrong_query, xq_vec)
        end

        @testset "Vector API with deriv=1 works" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp_fused = cubic_interp_fused(x, [y1, y2])
            mitp_comp = cubic_interp(x, [y1, y2])

            xq_vec = [0.2, 0.4, 0.6, 0.8]

            d1_fused = mitp_fused(xq_vec; deriv=1)
            d1_comp = mitp_comp(xq_vec; deriv=1)

            for k in 1:2
                @test d1_fused[k] ≈ d1_comp[k] atol=1e-14
            end
        end

        @testset "Type promotion with Integer query points" begin
            x = collect(range(0.0, 10.0, 101))
            y1 = sin.(π .* x ./ 10)
            y2 = cos.(π .* x ./ 10)

            mitp = cubic_interp_fused(x, [y1, y2])
            xq_int = [1, 3, 5, 7, 9]  # Integer query points

            outputs = mitp(xq_int)
            @test outputs isa Vector{Vector{Float64}}
            @test length(outputs) == 2
        end

    end  # Phase 7 testset

    # ========================================
    # Phase 8: Pre-built Anchor API
    # ========================================
    @testset "Phase 8: Pre-built Anchor API" begin

        @testset "Can build anchor vector via _fill_anchors!" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])
            xq_vec = [0.1, 0.3, 0.5, 0.7, 0.9]

            # Build anchors
            aq_vec = Vector{FastInterpolations._CubicAnchoredQuery{Float64}}(undef, length(xq_vec))
            FastInterpolations._fill_anchors!(aq_vec, mitp.x, xq_vec; wrap=false)

            @test length(aq_vec) == 5
            @test all(aq -> aq isa FastInterpolations._CubicAnchoredQuery{Float64}, aq_vec)
        end

        @testset "mitp(outputs, aq_vec) produces correct results" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            y3 = exp.(-x)

            mitp = cubic_interp_fused(x, [y1, y2, y3])
            xq_vec = [0.1, 0.3, 0.5, 0.7, 0.9]

            # Build anchors
            aq_vec = Vector{FastInterpolations._CubicAnchoredQuery{Float64}}(undef, length(xq_vec))
            FastInterpolations._fill_anchors!(aq_vec, mitp.x, xq_vec; wrap=false)

            # Evaluate with anchors
            outputs = [zeros(5) for _ in 1:3]
            result = mitp(outputs, aq_vec)

            @test result === outputs
            @test all(o -> any(v != 0.0 for v in o), outputs)  # Was filled
        end

        @testset "Anchor API results match query-point API exactly" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])
            xq_vec = [0.1, 0.25, 0.5, 0.75, 0.9]

            # Query-point API
            outputs_query = mitp(xq_vec)

            # Anchor API
            aq_vec = Vector{FastInterpolations._CubicAnchoredQuery{Float64}}(undef, length(xq_vec))
            FastInterpolations._fill_anchors!(aq_vec, mitp.x, xq_vec; wrap=false)
            outputs_anchor = [zeros(5) for _ in 1:2]
            mitp(outputs_anchor, aq_vec)

            # Results should be identical
            for k in 1:2
                @test outputs_anchor[k] ≈ outputs_query[k] atol=1e-14
            end
        end

        @testset "deriv keyword works with anchors" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])
            xq_vec = [0.2, 0.4, 0.6, 0.8]

            # Build anchors
            aq_vec = Vector{FastInterpolations._CubicAnchoredQuery{Float64}}(undef, length(xq_vec))
            FastInterpolations._fill_anchors!(aq_vec, mitp.x, xq_vec; wrap=false)

            # deriv=1 with anchors
            d1_query = mitp(xq_vec; deriv=1)
            d1_anchor = [zeros(4) for _ in 1:2]
            mitp(d1_anchor, aq_vec; deriv=1)

            for k in 1:2
                @test d1_anchor[k] ≈ d1_query[k] atol=1e-14
            end

            # deriv=2 with anchors
            d2_query = mitp(xq_vec; deriv=2)
            d2_anchor = [zeros(4) for _ in 1:2]
            mitp(d2_anchor, aq_vec; deriv=2)

            for k in 1:2
                @test d2_anchor[k] ≈ d2_query[k] atol=1e-14
            end
        end

        @testset "Validation for mismatched lengths" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])
            xq_vec = [0.1, 0.3, 0.5]

            # Build anchors
            aq_vec = Vector{FastInterpolations._CubicAnchoredQuery{Float64}}(undef, length(xq_vec))
            FastInterpolations._fill_anchors!(aq_vec, mitp.x, xq_vec; wrap=false)

            # Wrong number of series
            outputs_wrong_series = [zeros(3)]
            @test_throws DimensionMismatch mitp(outputs_wrong_series, aq_vec)

            # Wrong query length
            outputs_wrong_query = [zeros(5), zeros(5)]
            @test_throws DimensionMismatch mitp(outputs_wrong_query, aq_vec)
        end

        @testset "Anchors can be reused across multiple calls" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2])
            xq_vec = [0.2, 0.5, 0.8]

            # Build anchors once
            aq_vec = Vector{FastInterpolations._CubicAnchoredQuery{Float64}}(undef, length(xq_vec))
            FastInterpolations._fill_anchors!(aq_vec, mitp.x, xq_vec; wrap=false)

            # First call
            outputs1 = [zeros(3) for _ in 1:2]
            mitp(outputs1, aq_vec)

            # Second call with same anchors
            outputs2 = [zeros(3) for _ in 1:2]
            mitp(outputs2, aq_vec)

            # Results should be identical
            for k in 1:2
                @test outputs1[k] ≈ outputs2[k] atol=1e-14
            end
        end

        @testset "Wrap mode works with anchors" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)  # Periodic
            y2 = cos.(2π .* x)

            mitp = cubic_interp_fused(x, [y1, y2]; extrap=:wrap)
            xq_vec = [-0.1, 0.5, 1.3]  # Out-of-domain points

            # Build anchors with wrap=true
            aq_vec = Vector{FastInterpolations._CubicAnchoredQuery{Float64}}(undef, length(xq_vec))
            FastInterpolations._fill_anchors!(aq_vec, mitp.x, xq_vec; wrap=true)

            # Evaluate with anchors
            outputs_anchor = [zeros(3) for _ in 1:2]
            mitp(outputs_anchor, aq_vec)

            # Compare to query-point API
            outputs_query = mitp(xq_vec)

            for k in 1:2
                @test outputs_anchor[k] ≈ outputs_query[k] atol=1e-14
            end
        end

    end  # Phase 8 testset

    # ========================================
    # Phase 9: Matrix Input Constructor
    # ========================================
    @testset "Phase 9: Matrix Input Constructor" begin

        @testset "layout=:columns (default) - Y is n_points × n_series" begin
            x = collect(range(0.0, 1.0, 21))
            n_points = length(x)
            n_series = 3

            # Y matrix: each column is a series
            Y = Matrix{Float64}(undef, n_points, n_series)
            Y[:, 1] = sin.(2π .* x)
            Y[:, 2] = cos.(2π .* x)
            Y[:, 3] = exp.(-x)

            mitp = cubic_interp_fused(x, Y)  # default layout=:columns

            @test mitp isa CubicMultiInterpolantFused
            @test mitp.n_series == 3
            @test mitp.n_points == n_points
        end

        @testset "layout=:series_first - Y is n_series × n_points" begin
            x = collect(range(0.0, 1.0, 21))
            n_points = length(x)
            n_series = 3

            # Y matrix: each row is a series (same as internal storage)
            Y = Matrix{Float64}(undef, n_series, n_points)
            Y[1, :] = sin.(2π .* x)
            Y[2, :] = cos.(2π .* x)
            Y[3, :] = exp.(-x)

            mitp = cubic_interp_fused(x, Y; layout=:series_first)

            @test mitp isa CubicMultiInterpolantFused
            @test mitp.n_series == 3
            @test mitp.n_points == n_points
        end

        @testset "Both layouts produce equivalent results" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Column layout
            Y_cols = hcat(y1, y2)  # n_points × n_series
            mitp_cols = cubic_interp_fused(x, Y_cols; layout=:columns)

            # Row layout (series-first)
            Y_rows = vcat(y1', y2')  # n_series × n_points
            mitp_rows = cubic_interp_fused(x, Y_rows; layout=:series_first)

            # Results should be identical
            for xq in [0.1, 0.3, 0.5, 0.7, 0.9]
                @test mitp_cols(xq) ≈ mitp_rows(xq) atol=1e-14
            end
        end

        @testset "Matrix layout matches Vector{Vector} constructor" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Vector{Vector} API
            mitp_vec = cubic_interp_fused(x, [y1, y2])

            # Matrix API (columns)
            Y_cols = hcat(y1, y2)
            mitp_cols = cubic_interp_fused(x, Y_cols; layout=:columns)

            # Matrix API (series_first)
            Y_rows = vcat(y1', y2')
            mitp_rows = cubic_interp_fused(x, Y_rows; layout=:series_first)

            # All three should produce identical results
            for xq in [0.1, 0.5, 0.9]
                vals_vec = mitp_vec(xq)
                vals_cols = mitp_cols(xq)
                vals_rows = mitp_rows(xq)

                @test vals_cols ≈ vals_vec atol=1e-14
                @test vals_rows ≈ vals_vec atol=1e-14
            end
        end

        @testset "Invalid layout throws ArgumentError" begin
            x = collect(range(0.0, 1.0, 11))
            Y = ones(11, 2)

            @test_throws ArgumentError cubic_interp_fused(x, Y; layout=:invalid)
            @test_throws ArgumentError cubic_interp_fused(x, Y; layout=:row_major)
        end

        @testset "Dimension mismatch throws DimensionMismatch" begin
            x = collect(range(0.0, 1.0, 11))

            # Wrong number of rows for column layout
            Y_wrong = ones(10, 3)  # Should be 11 × n_series
            @test_throws DimensionMismatch cubic_interp_fused(x, Y_wrong; layout=:columns)

            # Wrong number of columns for series_first layout
            Y_wrong2 = ones(3, 10)  # Should be n_series × 11
            @test_throws DimensionMismatch cubic_interp_fused(x, Y_wrong2; layout=:series_first)
        end

        @testset "Works with all BC types" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            Y = hcat(y1, y2)

            # NaturalBC
            mitp_natural = cubic_interp_fused(x, Y; bc=NaturalBC())
            @test mitp_natural isa CubicMultiInterpolantFused

            # ClampedBC
            mitp_clamped = cubic_interp_fused(x, Y; bc=ClampedBC())
            @test mitp_clamped isa CubicMultiInterpolantFused

            # PeriodicBC (y1 and y2 are periodic)
            mitp_periodic = cubic_interp_fused(x, Y; bc=PeriodicBC())
            @test mitp_periodic isa CubicMultiInterpolantFused
        end

        @testset "Works with extrap keyword" begin
            x = collect(range(0.0, 1.0, 21))
            Y = hcat(sin.(2π .* x), cos.(2π .* x))

            for extrap in [:none, :constant, :extension, :wrap]
                mitp = cubic_interp_fused(x, Y; extrap=extrap)
                @test mitp.extrap === Val(extrap)
            end
        end

        @testset "Float32 matrix works" begin
            x = collect(range(0.0f0, 1.0f0, 21))
            Y = Matrix{Float32}(undef, 21, 2)
            Y[:, 1] = sin.(2π .* x)
            Y[:, 2] = cos.(2π .* x)

            mitp = cubic_interp_fused(x, Y)

            @test mitp isa CubicMultiInterpolantFused{Float32}
            vals = mitp(0.5f0)
            @test vals isa Vector{Float32}
        end

    end  # Phase 9 testset

    # ========================================
    # Phase 10: Conversion & Type Promotion
    # ========================================
    @testset "Phase 10: Conversion & Type Promotion" begin

        @testset "Conversion from CubicMultiInterpolant" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            y3 = exp.(-x)

            # Create composition-based interpolant
            mitp_comp = cubic_interp(x, [y1, y2, y3])

            # Convert to fused
            mitp_fused = CubicMultiInterpolantFused(mitp_comp)

            @test mitp_fused isa CubicMultiInterpolantFused{Float64}
            @test mitp_fused.n_series == 3
            @test mitp_fused.n_points == 51
        end

        @testset "Converted interpolant matches original results" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp_comp = cubic_interp(x, [y1, y2])
            mitp_fused = CubicMultiInterpolantFused(mitp_comp)

            # Results should be identical
            for xq in [0.1, 0.25, 0.5, 0.75, 0.9]
                vals_comp = mitp_comp(xq)
                vals_fused = mitp_fused(xq)
                @test vals_fused ≈ vals_comp atol=1e-14
            end
        end

        @testset "Conversion preserves BC settings" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Test with ClampedBC
            mitp_comp = cubic_interp(x, [y1, y2]; bc=ClampedBC())
            mitp_fused = CubicMultiInterpolantFused(mitp_comp)

            for xq in [0.2, 0.5, 0.8]
                @test mitp_fused(xq) ≈ mitp_comp(xq) atol=1e-14
            end
        end

        @testset "Conversion preserves extrap settings" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Test with :constant extrap
            mitp_comp = cubic_interp(x, [y1, y2]; extrap=:constant)
            mitp_fused = CubicMultiInterpolantFused(mitp_comp)

            @test mitp_fused.extrap === mitp_comp.itps[1].extrap

            # Out-of-bounds should match
            @test mitp_fused(1.5) ≈ mitp_comp(1.5) atol=1e-14
        end

        @testset "Vector{Vector} with Integer inputs works" begin
            x = 1:10  # Integer range
            y1 = [i^2 for i in 1:10]  # Integer vector
            y2 = [2*i for i in 1:10]

            mitp = cubic_interp_fused(x, [y1, y2])

            @test mitp isa CubicMultiInterpolantFused{Float64}
            vals = mitp(5.5)
            @test vals isa Vector{Float64}
        end

        @testset "Matrix with Integer inputs works" begin
            x = 1:10
            Y = [i^2 for i in 1:10, j in 1:3]  # 10×3 Integer matrix

            mitp = cubic_interp_fused(x, Y)

            @test mitp isa CubicMultiInterpolantFused{Float64}
        end

        @testset "Mixed type promotion works correctly" begin
            x = collect(range(0.0f0, 1.0f0, 21))  # Float32
            y1 = Float64.(sin.(2π .* x))  # Float64
            y2 = Float64.(cos.(2π .* x))

            # Should promote to Float64
            mitp = cubic_interp_fused(x, [y1, y2])
            @test mitp isa CubicMultiInterpolantFused{Float64}
        end

        @testset "Conversion works with derivatives" begin
            x = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            mitp_comp = cubic_interp(x, [y1, y2])
            mitp_fused = CubicMultiInterpolantFused(mitp_comp)

            # First derivative
            d1_comp = mitp_comp(0.5; deriv=1)
            d1_fused = mitp_fused(0.5; deriv=1)
            @test d1_fused ≈ d1_comp atol=1e-14

            # Second derivative
            d2_comp = mitp_comp(0.5; deriv=2)
            d2_fused = mitp_fused(0.5; deriv=2)
            @test d2_fused ≈ d2_comp atol=1e-14
        end

    end  # Phase 10 testset

end  # Main testset
