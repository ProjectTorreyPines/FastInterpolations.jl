# Cross-validation of PCHIP, Akima, and Cardinal against DataInterpolations.jl
# Verifies that FastInterpolations produces numerically equivalent results
# to the reference implementations.

using Test
using FastInterpolations
import DataInterpolations as DI
using Random

@testset "Local Slope Methods — Package Comparison" begin

    # ========================================
    # Test configurations
    # ========================================
    # Smooth function
    f_smooth(x) = sin(2π * x) + 0.5 * cos(4π * x)

    # Dense uniform grid
    x_uniform = collect(range(0.0, 1.0, 51))
    y_uniform = f_smooth.(x_uniform)

    # Non-uniform grid
    Random.seed!(42)
    x_nonuniform = sort(vcat(0.0, rand(48), 1.0))
    y_nonuniform = f_smooth.(x_nonuniform)

    # Interior query points (avoid exact grid points for cleaner comparison)
    xq = [0.03, 0.12, 0.27, 0.41, 0.55, 0.68, 0.79, 0.88, 0.95]

    # Monotone data (for PCHIP monotonicity comparison)
    x_mono = collect(range(0.0, 5.0, 10))
    y_mono = cumsum(rand(MersenneTwister(123), 10))

    # ========================================
    # PCHIP vs DataInterpolations.PCHIPInterpolation
    # ========================================
    @testset "PCHIP vs DataInterpolations" begin
        @testset "Uniform grid — interior points" begin
            di = DI.PCHIPInterpolation(y_uniform, x_uniform)
            for q in xq
                fi_val = pchip_interp(x_uniform, y_uniform, q)
                di_val = di(q)
                @test fi_val ≈ di_val rtol = 1e-12
            end
        end

        @testset "Non-uniform grid — interior points" begin
            di = DI.PCHIPInterpolation(y_nonuniform, x_nonuniform)
            for q in xq
                fi_val = pchip_interp(x_nonuniform, y_nonuniform, q)
                di_val = di(q)
                @test fi_val ≈ di_val rtol = 1e-12
            end
        end

        @testset "Monotone data — values match" begin
            di = DI.PCHIPInterpolation(y_mono, x_mono)
            xq_mono = collect(range(first(x_mono) + 0.1, last(x_mono) - 0.1, 20))
            for q in xq_mono
                fi_val = pchip_interp(x_mono, y_mono, q)
                di_val = di(q)
                @test fi_val ≈ di_val rtol = 1e-12
            end
        end

        @testset "Vector query — batch comparison" begin
            di = DI.PCHIPInterpolation(y_uniform, x_uniform)
            fi_vals = pchip_interp(x_uniform, y_uniform, xq)
            di_vals = [di(q) for q in xq]
            @test fi_vals ≈ di_vals rtol = 1e-12
        end
    end

    # ========================================
    # Akima vs DataInterpolations.AkimaInterpolation
    # ========================================
    @testset "Akima vs DataInterpolations" begin
        # Note: Akima boundary handling (virtual secant extrapolation) varies
        # between implementations. We use a generous tolerance for points near
        # boundaries and tight tolerance for interior points well away from edges.

        @testset "Uniform grid — interior points" begin
            di = DI.AkimaInterpolation(y_uniform, x_uniform)
            # Use points well away from boundaries
            xq_interior = [0.12, 0.27, 0.41, 0.55, 0.68, 0.79, 0.88]
            for q in xq_interior
                fi_val = akima_interp(x_uniform, y_uniform, q)
                di_val = di(q)
                @test fi_val ≈ di_val rtol = 1e-10
            end
        end

        @testset "Non-uniform grid — interior points" begin
            di = DI.AkimaInterpolation(y_nonuniform, x_nonuniform)
            xq_interior = [0.12, 0.27, 0.41, 0.55, 0.68, 0.79, 0.88]
            for q in xq_interior
                fi_val = akima_interp(x_nonuniform, y_nonuniform, q)
                di_val = di(q)
                @test fi_val ≈ di_val rtol = 1e-10
            end
        end

        @testset "Boundary points — looser tolerance" begin
            # Near-boundary points may differ due to virtual secant extrapolation
            di = DI.AkimaInterpolation(y_uniform, x_uniform)
            for q in [0.03, 0.95]
                fi_val = akima_interp(x_uniform, y_uniform, q)
                di_val = di(q)
                @test fi_val ≈ di_val atol = 0.01
            end
        end

        @testset "Vector query — batch comparison" begin
            di = DI.AkimaInterpolation(y_uniform, x_uniform)
            xq_safe = [0.12, 0.27, 0.41, 0.55, 0.68, 0.79, 0.88]
            fi_vals = akima_interp(x_uniform, y_uniform, xq_safe)
            di_vals = [di(q) for q in xq_safe]
            @test fi_vals ≈ di_vals rtol = 1e-10
        end
    end

    # ========================================
    # Cardinal (CatmullRom) — no DI equivalent, self-consistency check
    # ========================================
    @testset "Cardinal (CatmullRom) — self-consistency" begin
        # CatmullRom (tension=0): slopes = (y[k+1] - y[k-1]) / (x[k+1] - x[k-1])
        # Verify against manually computed slopes + Hermite
        x = collect(range(0.0, 1.0, 20))
        y = f_smooth.(x)

        # Manually compute CatmullRom slopes
        n = length(x)
        dy_manual = similar(y)
        dy_manual[1] = (y[2] - y[1]) / (x[2] - x[1])
        for k in 2:(n - 1)
            dy_manual[k] = (y[k + 1] - y[k - 1]) / (x[k + 1] - x[k - 1])
        end
        dy_manual[n] = (y[n] - y[n - 1]) / (x[n] - x[n - 1])

        # Cardinal with tension=0 should match Hermite with manual slopes
        for q in xq
            cardinal_val = cardinal_interp(x, y, q; tension = 0.0)
            hermite_val = cubic_interp(x, Hermite(y, dy_manual), q)
            @test cardinal_val ≈ hermite_val rtol = 1e-14
        end
    end

    # ========================================
    # Cross-method sanity: all methods should be close on smooth data
    # ========================================
    @testset "All methods — smooth data agreement" begin
        x = collect(range(0.0, 1.0, 51))
        y = f_smooth.(x)
        xq_test = [0.1, 0.3, 0.5, 0.7, 0.9]
        ref = f_smooth.(xq_test)

        spline = cubic_interp(x, y, xq_test)
        pchip = pchip_interp(x, y, xq_test)
        cardinal = cardinal_interp(x, y, xq_test)
        akima = akima_interp(x, y, xq_test)

        # All should be close to the true function on a dense grid
        @test spline ≈ ref atol = 1e-4
        @test pchip ≈ ref atol = 1e-3
        @test cardinal ≈ ref atol = 1e-2
        @test akima ≈ ref atol = 1e-3

        # All methods should agree with each other within reasonable tolerance
        @test pchip ≈ akima atol = 1e-2
        @test pchip ≈ spline atol = 1e-2
    end
end
