using Test
using FastInterpolations
using Symbolics
import SymbolicUtils

# Gate mirrors the `@static if isdefined(SymbolicUtils, :TypeT)` guard in
# ext/FastInterpolationsSymbolicsExt.jl — keep the two predicates in lockstep.
# On the Symbolics 6 / SymbolicUtils 3 generation the extension compiles out
# as a no-op, so the symbolic-tracing tests cannot run; the else branch pins
# the no-op contract instead. A runtime `if` is sufficient here (test files
# are not precompiled); switch to `@static if` should Symbolics-7-only macros
# ever appear in the gated body below.
SYMBOLICS_7_API = isdefined(SymbolicUtils, :TypeT)

if SYMBOLICS_7_API
    # Function barrier: measure a compiled symbolic function on concrete types
    # (the testset loops bind abstractly typed locals).
    function _compiled_call_allocs(f, x)
        f(x)  # warmup
        return @allocated f(x)
    end

    @testset "Symbolics extension active" begin
        ext = Base.get_extension(FastInterpolations, :FastInterpolationsSymbolicsExt)
        @test ext isa Module
        @test isdefined(ext, :DifferentiatedInterpolant)
        @test isdefined(ext, :DifferentiatedInterpolantND)
    end

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
            t_val = 0.3

            # First derivative: expand D(itp(t)) for every registered 1D family
            for (name, itp) in [
                    ("linear", linear_interp(x, y; extrap = ExtendExtrap())),
                    ("cubic", cubic_interp(x, y; extrap = ExtendExtrap())),
                    ("constant", constant_interp(x, y; extrap = ExtendExtrap())),
                    ("quadratic", quadratic_interp(x, y; extrap = ExtendExtrap())),
                ]
                @testset "$name" begin
                    dexpr = expand_derivatives(D(itp(t)))
                    @test dexpr isa Num

                    # Compile derivative and compare to numeric
                    df = build_function(dexpr, t; expression = Val{false})
                    @test df(t_val) ≈ itp(t_val; deriv = DerivOp(1))

                    # The derivative order is static, so the compiled call does not allocate
                    @test _compiled_call_allocs(df, t_val) <= ALLOC_THRESHOLD
                end
            end

            # Cubic spline (supports up to 3rd derivative)
            itp = cubic_interp(x, y; extrap = ExtendExtrap())
            expr = itp(t)

            # Second derivative
            d2expr = expand_derivatives(D(D(expr)))
            @test d2expr isa Num

            d2f = build_function(d2expr, t; expression = Val{false})
            @test d2f(t_val) ≈ itp(t_val; deriv = DerivOp(2))

            # Third derivative: orders keep accumulating
            d3expr = expand_derivatives(D(D(D(expr))))
            @test d3expr isa Num

            d3f = build_function(d3expr, t; expression = Val{false})
            @test d3f(t_val) ≈ itp(t_val; deriv = DerivOp(3))

            # Chain rule through the query: d/dt itp(2t) = 2 itp'(2t)
            cexpr = expand_derivatives(D(itp(2t)))
            @test cexpr isa Num

            cf = build_function(cexpr, t; expression = Val{false})
            @test cf(t_val) ≈ 2 * itp(2t_val; deriv = DerivOp(1))
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

        # ========================================
        # Registration hook contracts (direct invocation)
        # Pins the defensive promote_symtype/promote_shape overloads that the
        # Symbolics term machinery only invokes on non-default tracing paths.
        # ========================================
        @testset "registration hook contracts" begin
            ext = Base.get_extension(FastInterpolations, :FastInterpolationsSymbolicsExt)
            scalar = SymbolicUtils.ShapeVecT()

            x = collect(range(0.0, 1.0, 11))
            y = sin.(2π .* x)
            itp1 = cubic_interp(x, y; extrap = ExtendExtrap())
            d1 = ext.DifferentiatedInterpolant(itp1, 1)

            xg = range(0.0, 1.0, 11)
            data = [sin(xi) * cos(yj) for xi in xg, yj in xg]
            itpN = cubic_interp((xg, xg), data; extrap = ExtendExtrap())
            d = ext.DifferentiatedInterpolantND(itpN, (1, 0))

            # 1D hooks (plain + differentiated)
            @test SymbolicUtils.promote_symtype(itp1, Float64) === Real
            @test SymbolicUtils.promote_shape(itp1, scalar) == scalar
            @test SymbolicUtils.promote_symtype(d1, Float64) === Real
            @test SymbolicUtils.promote_shape(d1, scalar) == scalar

            # DifferentiatedInterpolant identity + numeric callable + compact display
            @test Base.nameof(d1) === :DifferentiatedFastInterpolation
            @test d1(0.3) ≈ itp1(0.3; deriv = DerivOp(1))
            @test sprint(show, d1) == "DifferentiatedInterpolant(" * sprint(show, itp1) * ", 1)"

            # ND hooks (plain + differentiated)
            @test SymbolicUtils.promote_symtype(itpN, Float64, Float64) === Real
            @test SymbolicUtils.promote_shape(itpN, scalar, scalar) == scalar
            @test SymbolicUtils.promote_symtype(d, Float64, Float64) === Real
            @test SymbolicUtils.promote_shape(d, scalar, scalar) == scalar

            # DifferentiatedInterpolantND identity + numeric callable
            @test Base.nameof(d) === :DifferentiatedFastInterpolationND
            @test d(0.3, 0.7) ≈ itpN((0.3, 0.7); deriv = DerivOp(1, 0))
        end
    end
else
    @info "Symbolics $(pkgversion(Symbolics)) / SymbolicUtils $(pkgversion(SymbolicUtils)): " *
        "SymbolicUtils.TypeT absent, so FastInterpolationsSymbolicsExt is a no-op; " *
        "running no-op contract tests instead of symbolic-tracing tests."

    @testset "Symbolics 6 no-op extension contract" begin
        # Vector grids on purpose: with Range grids the symbolic-call failure
        # type changes from TypeError (boolean branch on a Num in the binary
        # search) to MethodError (unsafe_trunc(Int, ::Num) in the direct
        # range search).
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        data = [sin(xi) * cos(yj) for xi in x, yj in x]

        # Mirror of the constructor list in the "1D Symbolic Calling" testset
        # of the Symbolics-7 branch above — update both together.
        itps_1d = [
            ("linear", linear_interp(x, y; extrap = ExtendExtrap())),
            ("cubic", cubic_interp(x, y; extrap = ExtendExtrap())),
            ("constant", constant_interp(x, y; extrap = ExtendExtrap())),
            ("quadratic", quadratic_interp(x, y; extrap = ExtendExtrap())),
        ]
        itp_c = itps_1d[2][2]
        itpN = cubic_interp((x, x), data; extrap = ExtendExtrap())

        @testset "extension loads as an empty no-op module" begin
            ext = Base.get_extension(FastInterpolations, :FastInterpolationsSymbolicsExt)
            @test ext isa Module
            # Names defined only inside the extension's `@static if` block —
            # update in lockstep with ext/FastInterpolationsSymbolicsExt.jl:
            @test !isdefined(ext, :DifferentiatedInterpolantND)
            @test !isdefined(ext, :DifferentiatedInterpolant)
        end

        @testset "numeric core unaffected with Symbolics loaded" begin
            for (name, itp) in itps_1d
                @testset "$name" begin
                    val = itp(0.3)
                    @test val isa Float64
                    @test isfinite(val)
                end
            end
            @test itp_c(0.3) ≈ sin(2π * 0.3) atol = 1.0e-3
            @test itp_c(0.3; deriv = DerivOp(1)) isa Float64
            @test itpN((0.3, 0.7)) ≈ sin(0.3) * cos(0.7) atol = 1.0e-4
            # numeric ND varargs is provided by the core protocol, not the extension:
            @test itpN(0.3, 0.7) == itpN((0.3, 0.7))
        end

        @testset "symbolic calls throw instead of returning silent wrong results" begin
            @variables t u v
            for (name, itp) in itps_1d
                @testset "$name" begin
                    @test_throws TypeError itp(t)
                end
            end
            @test_throws TypeError itpN((u, v))
            @test_throws TypeError itpN(u, v)
            @test_throws TypeError itpN(0.3, v)
            # the derivative pipeline fails at the inner symbolic call itp_c(t):
            @test_throws TypeError expand_derivatives(Differential(t)(itp_c(t)))
        end

        @testset "nameof not registered" begin
            @test_throws MethodError Base.nameof(itp_c)
            @test_throws MethodError Base.nameof(itpN)
        end
    end
end
