using Test
using FastInterpolations

@testset "coeffs / CellPoly" begin

    # ────────────────────────────────────────────
    # Test data
    # ────────────────────────────────────────────
    x_vec = collect(range(0.0, 3.0, length=31))
    x_nonuniform = [0.0, 0.3, 0.8, 1.5, 2.0, 2.7, 3.0]

    # Polynomial that cubic can represent exactly
    y_cubic = @. x_vec^3 - 2x_vec^2 + x_vec + 1
    y_cubic_nu = @. x_nonuniform^3 - 2x_nonuniform^2 + x_nonuniform + 1

    # Quadratic polynomial
    y_quad = @. x_vec^2 + 3x_vec + 1
    y_quad_nu = @. x_nonuniform^2 + 3x_nonuniform + 1

    # Linear function
    y_lin = @. 2.5x_vec + 1.0
    y_lin_nu = @. 2.5x_nonuniform + 1.0

    # Constant data
    y_const = @. sin(x_vec)
    y_const_nu = @. sin(x_nonuniform)

    # ────────────────────────────────────────────
    # CellPoly struct basics
    # ────────────────────────────────────────────
    @testset "CellPoly struct" begin
        cell = CellPoly{3, Float64, Float64}((1.0, 2.0, 3.0), 0.0, 1.0)
        @test cell.p == (1.0, 2.0, 3.0)
        @test cell.xL == 0.0
        @test cell.xR == 1.0

        # Callable
        @test cell(0.5) ≈ evalpoly(0.5, (1.0, 2.0, 3.0))

        # evalpoly overload
        @test evalpoly(0.5, cell) ≈ evalpoly(0.5, (1.0, 2.0, 3.0))

        # With shift
        cell2 = CellPoly{2, Float64, Float64}((3.0, 2.0), 1.0, 2.0)
        @test cell2(1.5) ≈ 3.0 + 2.0 * 0.5  # u = 1.5 - 1.0 = 0.5
        @test evalpoly(1.5, cell2) ≈ 3.0 + 2.0 * 0.5

        # Show
        io = IOBuffer()
        show(io, cell)
        s = String(take!(io))
        @test occursin("deg=2", s)
        @test occursin("Float64", s)
    end

    # ────────────────────────────────────────────
    # Cubic coeffs
    # ────────────────────────────────────────────
    @testset "cubic coeffs" begin
        itp = cubic_interp(x_vec, y_cubic)

        @testset "round-trip at random points" begin
            for xq in [0.05, 0.5, 1.0, 1.73, 2.5, 2.99]
                cell = coeffs(itp, xq)
                @test cell(xq) ≈ itp(xq) atol=1e-12
                @test evalpoly(xq, cell) ≈ itp(xq) atol=1e-12
            end
        end

        @testset "non-uniform grid" begin
            itp_nu = cubic_interp(x_nonuniform, y_cubic_nu)
            for xq in [0.1, 0.5, 1.2, 2.3, 2.9]
                cell = coeffs(itp_nu, xq)
                @test cell(xq) ≈ itp_nu(xq) atol=1e-12
            end
        end

        @testset "derivative from coefficients" begin
            # S(u) = d + c*u + b*u² + a*u³
            # S'(u) = c + 2b*u + 3a*u²
            for xq in [0.3, 1.5, 2.7]
                cell = coeffs(itp, xq)
                d, c, b, a = cell.p
                u = xq - cell.xL
                deriv_from_coeffs = c + 2b * u + 3a * u^2
                @test deriv_from_coeffs ≈ itp(xq; deriv=DerivOp(1)) atol=1e-10
            end
        end

        @testset "integration from coefficients" begin
            # ∫₀ʰ S(u) du = d*h + c/2*h² + b/3*h³ + a/4*h⁴
            for i in 1:5
                xL = x_vec[i]
                xR = x_vec[i+1]
                cell = coeffs(itp, xL + 0.001)
                d, c, b, a = cell.p
                h = cell.xR - cell.xL
                integral_from_coeffs = evalpoly(h, (zero(d), d, c/2, b/3, a/4))
                @test integral_from_coeffs ≈ integrate(itp, xL, xR) atol=1e-12
            end
        end

        @testset "cell boundary continuity" begin
            # At a shared grid point, both adjacent cells should give same value
            for i in 2:(length(x_vec)-1)
                xmid = x_vec[i]
                cell_left = coeffs(itp, xmid - eps(xmid))
                cell_right = coeffs(itp, xmid + eps(xmid))
                @test cell_left(xmid) ≈ cell_right(xmid) atol=1e-10
            end
        end

        @testset "returns CellPoly{4}" begin
            cell = coeffs(itp, 1.0)
            @test cell isa CellPoly{4, Float64, Float64}
        end

        @testset "search keyword" begin
            cell1 = coeffs(itp, 1.5; search=Binary())
            hint = Ref(1)
            cell2 = coeffs(itp, 1.5; search=LinearBinary(linear_window=0), hint=hint)
            @test cell1.p == cell2.p
            @test cell1.xL == cell2.xL
        end
    end

    # ────────────────────────────────────────────
    # Quadratic coeffs
    # ────────────────────────────────────────────
    @testset "quadratic coeffs" begin
        itp = quadratic_interp(x_vec, y_quad)

        @testset "round-trip" begin
            for xq in [0.05, 0.5, 1.5, 2.5, 2.99]
                cell = coeffs(itp, xq)
                @test cell(xq) ≈ itp(xq) atol=1e-10
                @test evalpoly(xq, cell) ≈ itp(xq) atol=1e-10
            end
        end

        @testset "non-uniform grid" begin
            itp_nu = quadratic_interp(x_nonuniform, y_quad_nu)
            for xq in [0.1, 0.5, 1.2, 2.3, 2.9]
                cell = coeffs(itp_nu, xq)
                @test cell(xq) ≈ itp_nu(xq) atol=1e-10
            end
        end

        @testset "returns CellPoly{3}" begin
            cell = coeffs(itp, 1.0)
            @test cell isa CellPoly{3, Float64, Float64}
        end

        @testset "integration from coefficients" begin
            for i in 1:5
                xL = x_vec[i]
                xR = x_vec[i+1]
                cell = coeffs(itp, xL + 0.001)
                y0, d, a = cell.p
                h = cell.xR - cell.xL
                integral_from_coeffs = evalpoly(h, (zero(y0), y0, d/2, a/3))
                @test integral_from_coeffs ≈ integrate(itp, xL, xR) atol=1e-12
            end
        end
    end

    # ────────────────────────────────────────────
    # Linear coeffs
    # ────────────────────────────────────────────
    @testset "linear coeffs" begin
        itp = linear_interp(x_vec, y_lin)

        @testset "round-trip" begin
            for xq in [0.05, 0.5, 1.5, 2.5, 2.99]
                cell = coeffs(itp, xq)
                @test cell(xq) ≈ itp(xq) atol=1e-12
                @test evalpoly(xq, cell) ≈ itp(xq) atol=1e-12
            end
        end

        @testset "non-uniform grid" begin
            itp_nu = linear_interp(x_nonuniform, y_lin_nu)
            for xq in [0.1, 0.5, 1.2, 2.3, 2.9]
                cell = coeffs(itp_nu, xq)
                @test cell(xq) ≈ itp_nu(xq) atol=1e-12
            end
        end

        @testset "returns CellPoly{2}" begin
            cell = coeffs(itp, 1.0)
            @test cell isa CellPoly{2, Float64, Float64}
        end

        @testset "slope matches" begin
            cell = coeffs(itp, 1.0)
            y0, slope = cell.p
            @test slope ≈ 2.5 atol=1e-12
        end

        @testset "integration from coefficients" begin
            for i in 1:5
                xL = x_vec[i]
                xR = x_vec[i+1]
                cell = coeffs(itp, xL + 0.001)
                y0, slope = cell.p
                h = cell.xR - cell.xL
                integral_from_coeffs = y0 * h + slope / 2 * h^2
                @test integral_from_coeffs ≈ integrate(itp, xL, xR) atol=1e-12
            end
        end
    end

    # ────────────────────────────────────────────
    # Constant coeffs
    # ────────────────────────────────────────────
    @testset "constant coeffs" begin
        @testset "side=LeftSide()" begin
            itp = constant_interp(x_vec, y_const; side=LeftSide())
            for xq in [0.05, 0.5, 1.5, 2.5]
                cell = coeffs(itp, xq)
                @test cell(xq) ≈ itp(xq) atol=1e-12
                @test cell isa CellPoly{1, Float64, Float64}
            end
        end

        @testset "side=RightSide()" begin
            itp = constant_interp(x_vec, y_const; side=RightSide())
            for xq in [0.05, 0.5, 1.5, 2.5]
                cell = coeffs(itp, xq)
                @test cell(xq) ≈ itp(xq) atol=1e-12
            end
        end

        @testset "side=NearestSide()" begin
            itp = constant_interp(x_vec, y_const; side=NearestSide())
            for xq in [0.05, 0.5, 1.5, 2.5]
                cell = coeffs(itp, xq)
                @test cell(xq) ≈ itp(xq) atol=1e-12
            end
        end
    end

    # ────────────────────────────────────────────
    # Complex value support
    # ────────────────────────────────────────────
    @testset "complex values" begin
        x_c = collect(range(0.0, 2.0, length=11))
        y_c = @. (1.0 + 2.0im) * x_c^2 + (3.0 - 1.0im)
        itp = cubic_interp(x_c, y_c)
        cell = coeffs(itp, 0.5)
        @test cell isa CellPoly{4, ComplexF64, Float64}
        @test cell(0.5) ≈ itp(0.5) atol=1e-10
    end

    # ────────────────────────────────────────────
    # Grid point queries
    # ────────────────────────────────────────────
    @testset "query at grid points" begin
        itp = cubic_interp(x_vec, y_cubic)

        # At interior grid points
        for i in 2:(length(x_vec)-1)
            cell = coeffs(itp, x_vec[i])
            @test cell(x_vec[i]) ≈ itp(x_vec[i]) atol=1e-12
        end

        # At first grid point
        cell_first = coeffs(itp, x_vec[1])
        @test cell_first(x_vec[1]) ≈ itp(x_vec[1]) atol=1e-12

        # At last grid point (should resolve to last cell)
        cell_last = coeffs(itp, x_vec[end])
        @test cell_last(x_vec[end]) ≈ itp(x_vec[end]) atol=1e-12
    end

end
