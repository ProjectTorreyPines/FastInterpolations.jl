using Test
using FastInterpolations
using Symbolics

@testset "Symbolics Registration" begin
    # ========================================
    # 1D Interpolant Registration
    # ========================================
    @testset "1D Symbolic Calling" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)

        @variables t

        for (name, itp) in [
                ("linear", linear_interp(x, y; extrap = ExtendExtrap())),
                ("cubic", cubic_interp(x, y; extrap = ExtendExtrap())),
                ("constant", constant_interp(x, y; extrap = ExtendExtrap())),
                ("quadratic", quadratic_interp(x, y; extrap = ExtendExtrap())),
            ]
            @testset "$name" begin
                # Symbolic expression creation
                expr = itp(t)
                @test expr isa Num

                # Compile to function and evaluate: symbolic roundtrip matches numeric
                f = build_function(expr, t; expression = Val{false})
                t_val = 0.3
                numeric_val = itp(t_val)
                compiled_val = f(t_val)
                @test compiled_val ≈ numeric_val
            end
        end
    end

    # ========================================
    # 1D Derivative Chain Rules
    # ========================================
    @testset "1D Symbolic Derivatives" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)

        @variables t
        D = Differential(t)

        # Cubic spline (supports up to 3rd derivative)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())

        # First derivative: expand D(itp(t))
        expr = itp(t)
        dexpr = expand_derivatives(D(expr))
        @test dexpr isa Num

        # Compile derivative and compare to numeric
        df = build_function(dexpr, t; expression = Val{false})
        t_val = 0.3
        numeric_deriv = itp(t_val; deriv = DerivOp(1))
        compiled_deriv = df(t_val)
        @test compiled_deriv ≈ numeric_deriv

        # Second derivative
        d2expr = expand_derivatives(D(D(expr)))
        @test d2expr isa Num

        d2f = build_function(d2expr, t; expression = Val{false})
        numeric_d2 = itp(t_val; deriv = DerivOp(2))
        compiled_d2 = d2f(t_val)
        @test compiled_d2 ≈ numeric_d2
    end

    # ========================================
    # ND Interpolant Registration
    # ========================================
    @testset "ND Symbolic Calling" begin
        xg = range(0.0, 1.0, 11)
        yg = range(0.0, 1.0, 11)
        data = [sin(xi) * cos(yj) for xi in xg, yj in yg]

        @variables u v
        itp = cubic_interp((xg, yg), data; extrap = ExtendExtrap())

        # Symbolic expression via tuple of Num
        expr = itp((u, v))
        @test expr isa Num

        # Compile and evaluate
        f = build_function(expr, [u, v]; expression = Val{false})
        u_val, v_val = 0.3, 0.7
        numeric_val = itp((u_val, v_val))
        compiled_val = f([u_val, v_val])
        @test compiled_val ≈ numeric_val
    end

    # ========================================
    # ND Symbolic Derivatives
    # ========================================
    @testset "ND Symbolic Derivatives" begin
        xg = range(0.0, 1.0, 21)
        yg = range(0.0, 1.0, 21)
        data = [sin(xi) * cos(yj) for xi in xg, yj in yg]

        @variables u v
        Du = Differential(u)
        Dv = Differential(v)
        itp = cubic_interp((xg, yg), data; extrap = ExtendExtrap())

        # Create symbolic expression
        expr = itp((u, v))

        # Partial derivative w.r.t. u
        du_expr = expand_derivatives(Du(expr))
        @test du_expr isa Num

        du_f = build_function(du_expr, [u, v]; expression = Val{false})
        u_val, v_val = 0.3, 0.7
        numeric_du = itp((u_val, v_val); deriv = DerivOp(1, 0))
        compiled_du = du_f([u_val, v_val])
        @test compiled_du ≈ numeric_du

        # Partial derivative w.r.t. v
        dv_expr = expand_derivatives(Dv(expr))
        @test dv_expr isa Num

        dv_f = build_function(dv_expr, [u, v]; expression = Val{false})
        numeric_dv = itp((u_val, v_val); deriv = DerivOp(0, 1))
        compiled_dv = dv_f([u_val, v_val])
        @test compiled_dv ≈ numeric_dv
    end

    # ========================================
    # nameof registration (1D + ND)
    # ========================================
    @testset "nameof" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        itp1 = cubic_interp(x, y; extrap = ExtendExtrap())
        @test Base.nameof(itp1) === :FastInterpolation

        xg = range(0.0, 1.0, 11)
        yg = range(0.0, 1.0, 11)
        data = [sin(xi) * cos(yj) for xi in xg, yj in yg]
        itpN = cubic_interp((xg, yg), data; extrap = ExtendExtrap())
        @test Base.nameof(itpN) === :FastInterpolationND
    end

    # ========================================
    # ND varargs symbolic form: itp(u, v) (vs the tuple form itp((u, v)))
    # ========================================
    @testset "ND Symbolic Calling (varargs)" begin
        xg = range(0.0, 1.0, 11)
        yg = range(0.0, 1.0, 11)
        data = [sin(xi) * cos(yj) for xi in xg, yj in yg]

        @variables u v
        itp = cubic_interp((xg, yg), data; extrap = ExtendExtrap())

        expr = itp(u, v)
        @test expr isa Num

        f = build_function(expr, [u, v]; expression = Val{false})
        u_val, v_val = 0.3, 0.7
        @test f([u_val, v_val]) ≈ itp((u_val, v_val))
    end

    # ========================================
    # Higher-order / mixed ND derivatives
    # (exercises DifferentiatedInterpolantND accumulation + varargs path)
    # ========================================
    @testset "ND Symbolic Derivatives (higher-order)" begin
        xg = range(0.0, 1.0, 21)
        yg = range(0.0, 1.0, 21)
        data = [sin(xi) * cos(yj) for xi in xg, yj in yg]

        @variables u v
        Du = Differential(u)
        Dv = Differential(v)
        itp = cubic_interp((xg, yg), data; extrap = ExtendExtrap())

        expr = itp((u, v))
        u_val, v_val = 0.3, 0.7

        # Mixed second partial d²/dudv: accumulates orders on DifferentiatedInterpolantND
        duv_expr = expand_derivatives(Du(Dv(expr)))
        @test duv_expr isa Num
        duv_f = build_function(duv_expr, [u, v]; expression = Val{false})
        @test duv_f([u_val, v_val]) ≈ itp((u_val, v_val); deriv = DerivOp(1, 1))

        # Pure second partial d²/du²
        duu_expr = expand_derivatives(Du(Du(expr)))
        @test duu_expr isa Num
        duu_f = build_function(duu_expr, [u, v]; expression = Val{false})
        @test duu_f([u_val, v_val]) ≈ itp((u_val, v_val); deriv = DerivOp(2, 0))
    end
end
