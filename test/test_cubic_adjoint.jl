using Test
using LinearAlgebra: dot
using FastInterpolations

# ========================================
# Helper: Dot-product test for adjoint correctness
# ========================================
# The gold standard: ⟨W·f, ȳ⟩ = ⟨f, W^T·ȳ⟩

function dot_product_test(x, xq, f, y_bar; bc = CubicFit(), atol = 0, rtol = sqrt(eps(eltype(x))))
    itp = cubic_interp(x, f; bc = bc)
    adj = cubic_adjoint(x, xq; bc = bc)

    # The forward is affine: y = W·f + c, where c comes from non-zero BC values.
    # Subtract the constant offset to isolate the linear part W·f.
    f_zero = zeros(eltype(f), length(f))
    itp_zero = cubic_interp(x, f_zero; bc = bc)
    Wf = itp.(xq) .- itp_zero.(xq)  # linear part only

    WTy = adj(y_bar)        # adjoint: ȳ → f̄

    lhs = dot(Wf, y_bar)
    rhs = dot(f, WTy)
    return lhs, rhs, isapprox(lhs, rhs; atol = atol, rtol = rtol)
end

@testset "CubicAdjoint" begin
    # ========================================
    # Test data setup
    # ========================================
    n_grid = 50
    n_query = 30

    x_uniform = collect(range(0.0, 1.0, n_grid))
    x_nonuniform = cumsum(0.5 .+ rand(n_grid))
    x_nonuniform .= (x_nonuniform .- x_nonuniform[1]) ./ (x_nonuniform[end] - x_nonuniform[1])

    xq = sort(rand(n_query)) .* 0.98 .+ 0.01  # inside domain with margin
    f = randn(n_grid)
    y_bar = randn(n_query)

    # ========================================
    # Dot-product tests: all BC types
    # ========================================
    @testset "Dot-product test — $bc_name" for (bc_name, bc) in [
        ("CubicFit (default)", CubicFit()),
        ("ZeroCurvBC", ZeroCurvBC()),
        ("ZeroSlopeBC", ZeroSlopeBC()),
        ("Deriv1(0.5)", BCPair(Deriv1(0.5), Deriv1(-0.3))),
        ("Deriv2(0.0)", BCPair(Deriv2(0.0), Deriv2(0.0))),
        ("Deriv3(0.0)", BCPair(Deriv3(0.0), Deriv3(0.0))),
        ("LinearFit", LinearFit()),
        ("QuadraticFit", QuadraticFit()),
        ("Mixed: CubicFit+Deriv2", BCPair(CubicFit(), Deriv2(0.0))),
        ("Mixed: Deriv1+Deriv3", BCPair(Deriv1(0.0), Deriv3(0.0))),
    ]
        @testset "Uniform grid" begin
            lhs, rhs, ok = dot_product_test(x_uniform, xq, f, y_bar; bc = bc)
            @test ok
        end
        @testset "Non-uniform grid" begin
            lhs, rhs, ok = dot_product_test(x_nonuniform, xq, f, y_bar; bc = bc)
            @test ok
        end
    end

    # ========================================
    # In-place vs allocating
    # ========================================
    @testset "In-place == allocating" begin
        adj = cubic_adjoint(x_uniform, xq; bc = CubicFit())
        f_bar_oop = adj(y_bar)
        f_bar_ip = zeros(n_grid)
        adj(f_bar_ip, y_bar)
        @test f_bar_oop ≈ f_bar_ip
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = cubic_adjoint(x_uniform, xq)
        @test size(adj) == (n_grid, n_query)
        @test size(adj, 1) == n_grid
        @test size(adj, 2) == n_query
    end

    # ========================================
    # Type stability
    # ========================================
    @testset "Type stability — Float64" begin
        adj = cubic_adjoint(x_uniform, xq; bc = CubicFit())
        @test @inferred(adj(y_bar)) isa Vector{Float64}
    end

    @testset "Type stability — Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        yb32 = randn(Float32, n_query)
        adj32 = cubic_adjoint(x32, xq32; bc = CubicFit())
        @test @inferred(adj32(yb32)) isa Vector{Float32}
    end

    # ========================================
    # Symmetric A sanity check
    # ========================================
    @testset "Transpose solve == forward solve for symmetric A" begin
        # For Deriv1/PolyFit BCs, A is symmetric → both solves give same result
        adj = cubic_adjoint(x_uniform, xq; bc = CubicFit())
        thomas = adj.cache.thomas

        b1 = randn(n_grid)
        b2 = copy(b1)

        FastInterpolations._ldiv_tridiagonal_nopiv!(b1, thomas)
        FastInterpolations._ldiv_tridiagonal_transpose!(b2, thomas)

        @test b1 ≈ b2
    end

    # ========================================
    # PeriodicBC adjoint
    # ========================================
    @testset "PeriodicBC" begin
        # Periodic data: f must satisfy f[end]=f[1] for inclusive grids
        n_p = 40
        n_pq = 25

        # Inclusive uniform grid on [0, 2π]
        x_incl = collect(range(0.0, 2π, n_p + 1))       # n_p+1 points, x[end]=x[1]+period
        f_incl = sin.(x_incl)
        f_incl[end] = f_incl[1]                           # enforce exact periodicity
        xq_p = sort(rand(n_pq)) .* 2π
        yb_p = randn(n_pq)

        # Inclusive non-uniform grid on [0, 2π]
        # Build n_p interior-style points in (0, 2π), then bracket with 0 and 2π
        x_nu_inner = cumsum(0.5 .+ rand(n_p - 1))
        x_nu_inner .= x_nu_inner ./ x_nu_inner[end] .* 2π .* ((n_p - 1) / n_p)  # scale to (0, ~2π)
        x_nu = vcat(0.0, x_nu_inner, 2π)   # n_p+1 points: 0, ..., 2π
        f_nu = sin.(x_nu)
        f_nu[end] = f_nu[1]

        # Exclusive uniform grid on [0, 2π)
        x_excl = range(0.0, step = 2π / n_p, length = n_p)  # Range, n_p points
        f_excl = sin.(collect(x_excl))

        @testset "Dot-product — inclusive uniform" begin
            lhs, rhs, ok = dot_product_test(x_incl, xq_p, f_incl, yb_p; bc = PeriodicBC())
            @test ok
        end

        @testset "Dot-product — inclusive non-uniform" begin
            lhs, rhs, ok = dot_product_test(x_nu, xq_p, f_nu, yb_p; bc = PeriodicBC())
            @test ok
        end

        @testset "Dot-product — exclusive Range" begin
            lhs, rhs, ok = dot_product_test(x_excl, xq_p, f_excl, yb_p;
                bc = PeriodicBC(endpoint = :exclusive))
            @test ok
        end

        @testset "In-place == allocating — periodic" begin
            adj = cubic_adjoint(x_incl, xq_p; bc = PeriodicBC())
            fb_oop = adj(yb_p)
            fb_ip = zeros(n_p + 1)
            adj(fb_ip, yb_p)
            @test fb_oop ≈ fb_ip
        end

        @testset "In-place == allocating — exclusive" begin
            adj = cubic_adjoint(x_excl, xq_p; bc = PeriodicBC(endpoint = :exclusive))
            fb_oop = adj(yb_p)
            fb_ip = zeros(n_p)
            adj(fb_ip, yb_p)
            @test fb_oop ≈ fb_ip
        end

        @testset "size — inclusive" begin
            adj = cubic_adjoint(x_incl, xq_p; bc = PeriodicBC())
            @test size(adj) == (n_p + 1, n_pq)
            @test size(adj, 1) == n_p + 1
        end

        @testset "size — exclusive" begin
            adj = cubic_adjoint(x_excl, xq_p; bc = PeriodicBC(endpoint = :exclusive))
            @test size(adj) == (n_p, n_pq)
            @test size(adj, 1) == n_p
        end

        @testset "Float32 — periodic" begin
            x32 = Float32.(x_incl)
            xq32 = Float32.(xq_p)
            f32 = sin.(Float32.(x_incl))
            f32[end] = f32[1]
            yb32 = randn(Float32, n_pq)
            lhs, rhs, ok = dot_product_test(x32, xq32, f32, yb32;
                bc = PeriodicBC(), rtol = sqrt(eps(Float32)))
            @test ok
        end

        @testset "Minimum grid — periodic (4 points)" begin
            x4 = collect(range(0.0, 2π, 4))   # 3 intervals
            xq4 = [1.0, 3.0, 5.0]
            f4 = sin.(x4)
            f4[end] = f4[1]
            yb4 = randn(3)
            lhs, rhs, ok = dot_product_test(x4, xq4, f4, yb4; bc = PeriodicBC())
            @test ok
        end
    end

    # ========================================
    # Float32 support
    # ========================================
    @testset "Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        f32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)

        lhs, rhs, ok = dot_product_test(x32, xq32, f32, yb32; bc = CubicFit(), rtol = sqrt(eps(Float32)))
        @test ok
    end

    # ========================================
    # Edge cases: minimum grid sizes
    # ========================================
    @testset "Minimum grid — CubicFit (4 points)" begin
        x_min = collect(range(0.0, 1.0, 4))
        xq_min = [0.25, 0.5, 0.75]
        f_min = randn(4)
        yb_min = randn(3)
        lhs, rhs, ok = dot_product_test(x_min, xq_min, f_min, yb_min; bc = CubicFit())
        @test ok
    end

    @testset "Minimum grid — Deriv2 (2 points)" begin
        x_min = [0.0, 1.0]
        xq_min = [0.5]
        f_min = randn(2)
        yb_min = randn(1)
        lhs, rhs, ok = dot_product_test(x_min, xq_min, f_min, yb_min; bc = BCPair(Deriv2(0.0), Deriv2(0.0)))
        @test ok
    end
end
