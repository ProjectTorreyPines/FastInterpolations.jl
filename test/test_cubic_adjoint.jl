@testitem "CubicAdjoint: Dot-product & PeriodicBC" setup = [AllocConstants] begin
    using LinearAlgebra: dot

    # ========================================
    # Helper: Dot-product test for adjoint correctness
    # ========================================
    # The gold standard: ⟨W·f, ȳ⟩ = ⟨f, W^T·ȳ⟩

    function dot_product_test(
            x, xq, f, y_bar;
            bc = CubicFit(), extrap = NoExtrap(), deriv = EvalValue(),
            atol = 0, rtol = sqrt(eps(eltype(x)))
        )
        itp = cubic_interp(x, f; bc = bc, extrap = extrap)
        adj = cubic_adjoint(x, xq; bc = bc, extrap = extrap)

        # The forward is affine: y = W·f + c, where c comes from non-zero BC values.
        # Subtract the constant offset to isolate the linear part W_d·f.
        f_zero = zeros(eltype(f), length(f))
        itp_zero = cubic_interp(x, f_zero; bc = bc, extrap = extrap)
        Wf = itp.(xq; deriv = deriv) .- itp_zero.(xq; deriv = deriv)  # linear part only

        WTy = adj(y_bar; deriv = deriv)        # adjoint: ȳ → f̄

        lhs = dot(Wf, y_bar)
        rhs = dot(f, WTy)
        return lhs, rhs, isapprox(lhs, rhs; atol = atol, rtol = rtol)
    end


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
    # Dot-product tests: extrap modes with OOB queries
    # ========================================
    xq_oob = vcat(-0.3, -0.1, sort(rand(n_query)) .* 0.98 .+ 0.01, 1.1, 1.3)
    y_bar_oob = randn(length(xq_oob))

    @testset "Dot-product — OOB queries — ExtendExtrap" begin
        @testset "Uniform" begin
            _, _, ok = dot_product_test(x_uniform, xq_oob, f, y_bar_oob; extrap = ExtendExtrap())
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = dot_product_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = ExtendExtrap())
            @test ok
        end
    end

    @testset "Dot-product — OOB queries — ClampExtrap" begin
        @testset "Uniform" begin
            _, _, ok = dot_product_test(x_uniform, xq_oob, f, y_bar_oob; extrap = ClampExtrap())
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = dot_product_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = ClampExtrap())
            @test ok
        end
    end

    @testset "Dot-product — OOB queries — WrapExtrap" begin
        @testset "Uniform" begin
            _, _, ok = dot_product_test(x_uniform, xq_oob, f, y_bar_oob; extrap = WrapExtrap())
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = dot_product_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = WrapExtrap())
            @test ok
        end
    end

    @testset "Dot-product — OOB queries — FillExtrap" begin
        @testset "Uniform" begin
            _, _, ok = dot_product_test(x_uniform, xq_oob, f, y_bar_oob; extrap = FillExtrap(0.0))
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = dot_product_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = FillExtrap(0.0))
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
    # Type stability — constructor @inferred
    # ========================================
    @testset "Type stability — constructor @inferred" begin
        # BC variations
        @test @inferred(cubic_adjoint(x_uniform, xq)) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; bc = CubicFit())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; bc = ZeroCurvBC())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; bc = ZeroSlopeBC())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; bc = LinearFit())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; bc = QuadraticFit())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; bc = BCPair(Deriv1(0.0), Deriv2(0.0)))) isa CubicAdjoint

        # Periodic BC
        x_per = collect(range(0.0, 2π, n_grid))
        f_per = sin.(x_per)
        f_per[end] = f_per[1]
        @test @inferred(cubic_adjoint(x_per, xq; bc = PeriodicBC())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_per[1:(end - 1)], xq; bc = PeriodicBC(endpoint = :exclusive, period = 2π))) isa CubicAdjoint

        # Extrap variations
        @test @inferred(cubic_adjoint(x_uniform, xq; extrap = NoExtrap())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; extrap = ExtendExtrap())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; extrap = ClampExtrap())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; extrap = WrapExtrap())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; extrap = FillExtrap(0.0))) isa CubicAdjoint

        # BC + extrap combinations
        @test @inferred(cubic_adjoint(x_uniform, xq; bc = ZeroCurvBC(), extrap = ExtendExtrap())) isa CubicAdjoint
        @test @inferred(cubic_adjoint(x_uniform, xq; bc = CubicFit(), extrap = WrapExtrap())) isa CubicAdjoint
    end

    # ========================================
    # Type stability — apply @inferred
    # ========================================
    @testset "Type stability — apply Float64" begin
        adj = cubic_adjoint(x_uniform, xq; bc = CubicFit())
        @test @inferred(adj(y_bar)) isa Vector{Float64}
    end

    @testset "Type stability — apply Float32" begin
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
            lhs, rhs, ok = dot_product_test(
                x_excl, xq_p, f_excl, yb_p;
                bc = PeriodicBC(endpoint = :exclusive)
            )
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
            lhs, rhs, ok = dot_product_test(
                x32, xq32, f32, yb32;
                bc = PeriodicBC(), rtol = sqrt(eps(Float32))
            )
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

end

# ========================================
# Complex data support
# ========================================
@testitem "CubicAdjoint: Complex & Derivative" setup = [AllocConstants] begin
    using LinearAlgebra: dot

    function dot_product_test(
            x, xq, f, y_bar;
            bc = CubicFit(), extrap = NoExtrap(), deriv = EvalValue(),
            atol = 0, rtol = sqrt(eps(eltype(x)))
        )
        itp = cubic_interp(x, f; bc = bc, extrap = extrap)
        adj = cubic_adjoint(x, xq; bc = bc, extrap = extrap)
        f_zero = zeros(eltype(f), length(f))
        itp_zero = cubic_interp(x, f_zero; bc = bc, extrap = extrap)
        Wf = itp.(xq; deriv = deriv) .- itp_zero.(xq; deriv = deriv)
        WTy = adj(y_bar; deriv = deriv)
        lhs = dot(Wf, y_bar)
        rhs = dot(f, WTy)
        return lhs, rhs, isapprox(lhs, rhs; atol = atol, rtol = rtol)
    end

    n_grid = 50
    n_query = 30
    x_uniform = collect(range(0.0, 1.0, n_grid))
    x_nonuniform = cumsum(0.5 .+ rand(n_grid))
    x_nonuniform .= (x_nonuniform .- x_nonuniform[1]) ./ (x_nonuniform[end] - x_nonuniform[1])
    xq = sort(rand(n_query)) .* 0.98 .+ 0.01
    f = randn(n_grid)
    y_bar = randn(n_query)

    @testset "Complex — dot-product — $bc_name" for (bc_name, bc) in [
            ("CubicFit", CubicFit()),
            ("ZeroCurvBC", ZeroCurvBC()),
            ("ZeroSlopeBC", ZeroSlopeBC()),
            ("Deriv1(0.5)", BCPair(Deriv1(0.5), Deriv1(-0.3))),
            ("PeriodicBC", PeriodicBC()),
        ]
        f_c = randn(ComplexF64, n_grid)
        yb_c = randn(ComplexF64, n_query)

        if bc isa PeriodicBC
            x_p = collect(range(0.0, 2π, n_grid))
            f_p = randn(ComplexF64, n_grid)
            f_p[end] = f_p[1]
            xq_p = sort(rand(n_query)) .* 2π
            yb_p = randn(ComplexF64, n_query)
            lhs, rhs, ok = dot_product_test(x_p, xq_p, f_p, yb_p; bc = bc)
            @test ok
        else
            @testset "Uniform" begin
                lhs, rhs, ok = dot_product_test(x_uniform, xq, f_c, yb_c; bc = bc)
                @test ok
            end
            @testset "Non-uniform" begin
                lhs, rhs, ok = dot_product_test(x_nonuniform, xq, f_c, yb_c; bc = bc)
                @test ok
            end
        end
    end

    @testset "Complex — in-place == allocating" begin
        adj = cubic_adjoint(x_uniform, xq; bc = CubicFit())
        yb_c = randn(ComplexF64, n_query)
        fb_oop = adj(yb_c)
        fb_ip = zeros(ComplexF64, n_grid)
        adj(fb_ip, yb_c)
        @test fb_oop ≈ fb_ip
    end

    @testset "Complex — derivative adjoint — deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        f_c = randn(ComplexF64, n_grid)
        yb_c = randn(ComplexF64, n_query)
        _, _, ok = dot_product_test(x_uniform, xq, f_c, yb_c; bc = CubicFit(), deriv = op)
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

    # ========================================
    # Derivative adjoint: deriv keyword
    # ========================================
    @testset "Derivative adjoint — $d_name, $bc_name" for
        (d_name, deriv_op) in [
                ("deriv=1", EvalDeriv1()),
                ("deriv=2", EvalDeriv2()),
                ("deriv=3", EvalDeriv3()),
            ],
            (bc_name, bc) in [
                ("CubicFit", CubicFit()),
                ("ZeroCurvBC", ZeroCurvBC()),
                ("Deriv1(0.5)", BCPair(Deriv1(0.5), Deriv1(-0.3))),
                ("Mixed: CubicFit+Deriv2", BCPair(CubicFit(), Deriv2(0.0))),
            ]
        @testset "Uniform grid" begin
            _, _, ok = dot_product_test(x_uniform, xq, f, y_bar; bc = bc, deriv = deriv_op)
            @test ok
        end
        @testset "Non-uniform grid" begin
            _, _, ok = dot_product_test(x_nonuniform, xq, f, y_bar; bc = bc, deriv = deriv_op)
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: in-place == allocating
    # ========================================
    @testset "In-place == allocating — deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        adj = cubic_adjoint(x_uniform, xq; bc = CubicFit())
        fb_oop = adj(y_bar; deriv = op)
        fb_ip = zeros(n_grid)
        adj(fb_ip, y_bar; deriv = op)
        @test fb_oop ≈ fb_ip
    end

    # ========================================
    # Derivative adjoint: periodic BC
    # ========================================
    @testset "Derivative adjoint — periodic" begin
        n_p = 40
        n_pq = 25
        x_incl = collect(range(0.0, 2π, n_p + 1))
        f_incl = sin.(x_incl)
        f_incl[end] = f_incl[1]
        xq_p = sort(rand(n_pq)) .* 2π
        yb_p = randn(n_pq)

        x_excl = range(0.0, step = 2π / n_p, length = n_p)
        f_excl = sin.(collect(x_excl))

        @testset "$d_name — inclusive" for (d_name, op) in [
                ("deriv=1", DerivOp(1)), ("deriv=2", DerivOp(2)), ("deriv=3", DerivOp(3)), ("deriv=4", DerivOp(4)),
            ]
            _, _, ok = dot_product_test(
                x_incl, xq_p, f_incl, yb_p;
                bc = PeriodicBC(), deriv = op
            )
            @test ok
        end

        @testset "$d_name — exclusive" for (d_name, op) in [
                ("deriv=1", DerivOp(1)), ("deriv=2", DerivOp(2)), ("deriv=3", DerivOp(3)), ("deriv=4", DerivOp(4)),
            ]
            _, _, ok = dot_product_test(
                x_excl, xq_p, f_excl, yb_p;
                bc = PeriodicBC(endpoint = :exclusive), deriv = op
            )
            @test ok
        end
    end

    # ========================================
    # Float32 derivative adjoint
    # ========================================
    @testset "Float32 — deriv=1" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        f32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        _, _, ok = dot_product_test(
            x32, xq32, f32, yb32;
            bc = CubicFit(), deriv = EvalDeriv1(), rtol = sqrt(eps(Float32))
        )
        @test ok
    end

end

# ========================================
# Matrix materialization
# ========================================
@testitem "CubicAdjoint: Matrix & Zero-alloc & y_bar variants" setup = [AllocConstants] begin
    using LinearAlgebra: dot

    n_grid = 50
    n_query = 30
    x_uniform = collect(range(0.0, 1.0, n_grid))
    x_nonuniform = cumsum(0.5 .+ rand(n_grid))
    x_nonuniform .= (x_nonuniform .- x_nonuniform[1]) ./ (x_nonuniform[end] - x_nonuniform[1])
    xq = sort(rand(n_query)) .* 0.98 .+ 0.01
    f = randn(n_grid)
    y_bar = randn(n_query)

    @testset "Matrix(adj) — Wᵀ·ȳ == adj(ȳ)" begin
        adj = cubic_adjoint(x_uniform, xq; bc = CubicFit())
        W_T = Matrix(adj)
        @test size(W_T) == (n_grid, n_query)
        @test W_T * y_bar ≈ adj(y_bar)
    end

    @testset "Matrix(adj)' — W·f == itp.(xq)" begin
        itp = cubic_interp(x_uniform, f; bc = CubicFit())
        adj = cubic_adjoint(x_uniform, xq; bc = CubicFit())
        W = Matrix(adj)'
        @test W * f ≈ itp.(xq)
    end

    @testset "Matrix(itp, xq) — convenience forward matrix" begin
        itp = cubic_interp(x_uniform, f; bc = CubicFit())
        W = Matrix(itp, xq)
        @test size(W) == (n_query, n_grid)
        @test W * f ≈ itp.(xq)
    end

    @testset "Matrix — deriv keyword" begin
        adj = cubic_adjoint(x_uniform, xq; bc = CubicFit())
        itp = cubic_interp(x_uniform, f; bc = CubicFit())
        for (name, op) in [("d1", DerivOp(1)), ("d2", DerivOp(2)), ("d3", DerivOp(3)), ("d4", DerivOp(4))]
            W_T = Matrix(adj; deriv = op)
            @test W_T * y_bar ≈ adj(y_bar; deriv = op)
            W = Matrix(itp, xq; deriv = op)
            @test W * f ≈ itp.(xq; deriv = op)
        end
    end

    @testset "Matrix — non-uniform grid" begin
        adj = cubic_adjoint(x_nonuniform, xq; bc = CubicFit())
        itp = cubic_interp(x_nonuniform, f; bc = CubicFit())
        W_T = Matrix(adj)
        @test W_T * y_bar ≈ adj(y_bar)
        @test Matrix(adj)' * f ≈ itp.(xq)
    end

    @testset "Matrix — non-zero BC (affine)" begin
        bc = BCPair(Deriv1(0.5), Deriv1(-0.3))
        adj = cubic_adjoint(x_uniform, xq; bc = bc)
        itp = cubic_interp(x_uniform, f; bc = bc)
        # Affine: subtract constant to isolate linear part
        itp_zero = cubic_interp(x_uniform, zeros(n_grid); bc = bc)
        Wf = itp.(xq) .- itp_zero.(xq)
        @test Matrix(adj)' * f ≈ Wf
    end

    @testset "Matrix — periodic" begin
        n_p = 20
        x_p = collect(range(0.0, 2π, n_p + 1))
        f_p = sin.(x_p); f_p[end] = f_p[1]
        xq_p = sort(rand(15)) .* 2π
        yb_p = randn(15)

        adj = cubic_adjoint(x_p, xq_p; bc = PeriodicBC())
        @test Matrix(adj) * yb_p ≈ adj(yb_p)

        itp = cubic_interp(x_p, f_p; bc = PeriodicBC())
        @test Matrix(itp, xq_p; deriv = EvalValue()) * f_p ≈ itp.(xq_p)
    end

    @testset "Matrix — Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        adj32 = cubic_adjoint(x32, xq32; bc = CubicFit())
        W_T = Matrix(adj32)
        @test eltype(W_T) == Float32
        @test size(W_T) == (n_grid, n_query)
    end

    # ========================================
    # Allocation tests (zero-alloc in-place)
    # ========================================

    # Function barriers: @testset wraps body in try/catch → type-unstable locals.
    # All setup + warmup + @allocated must be inside ONE function for accurate results.

    function _test_adjoint_alloc_inplace(x, xq, f_bar, y_bar; bc = CubicFit(), deriv = EvalValue())
        adj = cubic_adjoint(x, xq; bc = bc)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place non-periodic" begin
        fb = zeros(n_grid)
        allocs = _test_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar; bc = CubicFit())
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place non-periodic (non-uniform)" begin
        fb = zeros(n_grid)
        allocs = _test_adjoint_alloc_inplace(x_nonuniform, xq, fb, y_bar; bc = CubicFit())
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place ZeroCurvBC" begin
        fb = zeros(n_grid)
        allocs = _test_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar; bc = ZeroCurvBC())
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place Deriv1(0.5)" begin
        fb = zeros(n_grid)
        allocs = _test_adjoint_alloc_inplace(
            x_uniform, xq, fb, y_bar; bc = BCPair(Deriv1(0.5), Deriv1(-0.3))
        )
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place periodic (inclusive)" begin
        n_p = 40
        x_p = collect(range(0.0, 2π, n_p + 1))
        xq_p = sort(rand(25)) .* 2π
        fb_p = zeros(n_p + 1)
        yb_p = randn(25)
        allocs = _test_adjoint_alloc_inplace(x_p, xq_p, fb_p, yb_p; bc = PeriodicBC())
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place periodic (exclusive)" begin
        n_p = 40
        x_p = range(0.0, step = 2π / n_p, length = n_p)
        xq_p = sort(rand(25)) .* 2π
        fb_p = zeros(n_p)
        yb_p = randn(25)
        allocs = _test_adjoint_alloc_inplace(
            x_p, xq_p, fb_p, yb_p; bc = PeriodicBC(endpoint = :exclusive)
        )
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        fb = zeros(n_grid)
        allocs = _test_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar; deriv = op)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        fb32 = zeros(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        allocs = _test_adjoint_alloc_inplace(x32, xq32, fb32, yb32; bc = CubicFit())
        @test allocs <= ALLOC_THRESHOLD
    end
end

# ========================================
# Scalar / Tuple y_bar + scalar query constructor
# ========================================
@testitem "CubicAdjoint scalar/tuple y_bar" setup = [AllocConstants] begin
    x = range(0.0, 1.0, 20)

    @testset "Scalar y_bar (1 query point)" begin
        xq_single = [0.3]
        adj = cubic_adjoint(x, xq_single)
        ref = adj([1.5])
        @test adj(1.5) ≈ ref
        @test adj((1.5,)) ≈ ref

        # In-place
        f_bar = zeros(20)
        adj(f_bar, 1.5)
        @test f_bar ≈ ref

        fill!(f_bar, 0.0)
        adj(f_bar, (1.5,))
        @test f_bar ≈ ref
    end

    @testset "Tuple y_bar (multiple query points)" begin
        xq = [0.2, 0.5, 0.8]
        adj = cubic_adjoint(x, xq)
        ref = adj([1.0, 2.0, 3.0])
        @test adj((1.0, 2.0, 3.0)) ≈ ref

        # In-place
        f_bar = zeros(20)
        adj(f_bar, (1.0, 2.0, 3.0))
        @test f_bar ≈ ref
    end

    @testset "Scalar y_bar with deriv" begin
        xq_single = [0.4]
        adj = cubic_adjoint(x, xq_single)
        ref = adj([1.0]; deriv = DerivOp(1))
        @test adj(1.0; deriv = DerivOp(1)) ≈ ref
    end

    @testset "Scalar query constructor" begin
        adj_vec = cubic_adjoint(x, [0.5])
        adj_scalar = cubic_adjoint(x, 0.5)
        @test adj_vec(1.0) ≈ adj_scalar(1.0)
    end

    @testset "Dimension mismatch errors" begin
        adj = cubic_adjoint(x, [0.2, 0.5])  # 2 queries
        @test_throws DimensionMismatch adj(1.5)         # scalar but 2 queries
        @test_throws DimensionMismatch adj((1.0,))       # 1-tuple but 2 queries
    end

    @testset "Periodic BC — scalar/tuple y_bar" begin
        for (bc, x_per) in [
                (PeriodicBC(), collect(range(0.0, 2π, 21))),
                (PeriodicBC(endpoint = :exclusive, period = 2π), collect(range(0.0; step = 2π / 20, length = 20))),
            ]
            xq = [1.0, 3.0, 5.0]
            adj = cubic_adjoint(x_per, xq; bc = bc)
            ref = adj([1.0, 2.0, 3.0])

            # Tuple y_bar
            @test adj((1.0, 2.0, 3.0)) ≈ ref

            # In-place tuple
            n_out = length(ref)
            f_bar = zeros(n_out)
            adj(f_bar, (1.0, 2.0, 3.0))
            @test f_bar ≈ ref

            # Scalar y_bar (single query)
            adj1 = cubic_adjoint(x_per, [2.0]; bc = bc)
            ref1 = adj1([1.5])
            @test adj1(1.5) ≈ ref1
            @test adj1((1.5,)) ≈ ref1

            # In-place scalar
            f_bar1 = zeros(length(ref1))
            adj1(f_bar1, 1.5)
            @test f_bar1 ≈ ref1
        end
    end
end
