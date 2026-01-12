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

end  # Main testset
