@testitem "Constant One-Shot Series" setup = [AllocConstants] begin
    x = collect(range(0.0, 1.0, 101))
    y_sin = sin.(2π .* x)
    y_cos = cos.(2π .* x)
    y_exp = exp.(x)

    sitp = constant_interp(x, Series(y_sin, y_cos, y_exp))

    @testset "Scalar: Tuple → NTuple" begin
        xq = 0.37
        vals = constant_interp(x, Series(y_sin, y_cos, y_exp), xq)
        ref = sitp(xq)
        @test vals isa Vector{Float64}
        @test vals ≈ ref
    end

    @testset "Scalar: Matrix → Vector" begin
        Y = hcat(y_sin, y_cos, y_exp)
        vals = constant_interp(x, Series(Y), 0.37)
        ref = sitp(0.37)
        @test vals isa Vector{Float64}
        @test vals ≈ ref
    end

    @testset "In-place scalar" begin
        out = zeros(3)
        ret = constant_interp!(out, x, Series(y_sin, y_cos, y_exp), 0.37)
        ref = sitp(0.37)
        @test ret === out
        @test out ≈ ref
    end

    @testset "Vector query" begin
        xqs = [0.1, 0.37, 0.5, 0.9]
        outs = constant_interp(x, Series(y_sin, y_cos, y_exp), xqs)
        @test length(outs) == 3
        for j in eachindex(xqs)
            ref = sitp(xqs[j])
            for k in 1:3
                @test outs[k][j] ≈ ref[k]
            end
        end
    end

    @testset "Side options" begin
        for side_opt in [NearestSide(), LeftSide(), RightSide()]
            sitp_s = constant_interp(x, Series(y_sin, y_cos); side = side_opt)
            vals = constant_interp(x, Series(y_sin, y_cos), 0.37; side = side_opt)
            ref = sitp_s(0.37)
            @test collect(vals) ≈ ref
        end
    end

    @testset "Extrapolation modes" begin
        xq_oob = 1.5
        @test_throws DomainError constant_interp(x, Series(y_sin, y_cos), xq_oob)

        vals_clamp = constant_interp(x, Series(y_sin, y_cos), xq_oob; extrap = ClampExtrap())
        ref_sin = constant_interp(x, y_sin, xq_oob; extrap = ClampExtrap())
        ref_cos = constant_interp(x, y_cos, xq_oob; extrap = ClampExtrap())
        @test vals_clamp[1] ≈ ref_sin
        @test vals_clamp[2] ≈ ref_cos

        vals_ext = constant_interp(x, Series(y_sin, y_cos), xq_oob; extrap = ExtendExtrap())
        ref_sin_ext = constant_interp(x, y_sin, xq_oob; extrap = ExtendExtrap())
        @test vals_ext[1] ≈ ref_sin_ext

        # WrapExtrap
        vals_wrap = constant_interp(x, Series(y_sin, y_cos), xq_oob; extrap = WrapExtrap())
        ref_wrap_sin = constant_interp(x, y_sin, xq_oob; extrap = WrapExtrap())
        ref_wrap_cos = constant_interp(x, y_cos, xq_oob; extrap = WrapExtrap())
        @test vals_wrap[1] ≈ ref_wrap_sin
        @test vals_wrap[2] ≈ ref_wrap_cos

        # FillExtrap
        vals_fill = constant_interp(x, Series(y_sin, y_cos), xq_oob; extrap = FillExtrap(999.0))
        @test vals_fill[1] ≈ 999.0
        @test vals_fill[2] ≈ 999.0
    end

    @testset "Derivative ops" begin
        for d in 0:2
            op = DerivOp(d)
            vals = constant_interp(x, Series(y_sin, y_cos), 0.37; deriv = op)
            ref_sin = constant_interp(x, y_sin, 0.37; deriv = op)
            ref_cos = constant_interp(x, y_cos, 0.37; deriv = op)
            @test vals[1] ≈ ref_sin
            @test vals[2] ≈ ref_cos
        end
    end

    @testset "Float32" begin
        x32 = Float32.(x)
        y32_sin = Float32.(y_sin)
        y32_cos = Float32.(y_cos)
        vals = constant_interp(x32, Series(y32_sin, y32_cos), 0.5f0)
        @test vals isa Vector{Float32}
        ref_sin = constant_interp(x32, y32_sin, 0.5f0)
        @test vals[1] ≈ ref_sin
    end

    @testset "Complex values" begin
        y_c1 = complex.(y_sin, y_cos)
        y_c2 = complex.(y_exp, -y_exp)
        vals = constant_interp(x, Series(y_c1, y_c2), 0.5)
        ref1 = constant_interp(x, y_c1, 0.5)
        ref2 = constant_interp(x, y_c2, 0.5)
        @test vals[1] ≈ ref1
        @test vals[2] ≈ ref2
    end

    # ForwardDiff AD: skipped for constant interp (piecewise constant is
    # discontinuous, and _ConstantAnchoredQuery uses Tg(xi_primal) which
    # does not support Dual types)

    # Function-barrier pattern: pass outer vars as args (avoids @testset
    # try/catch type-instability under @testitem fresh-module compilation).
    @testset "Zero allocation (in-place scalar)" begin
        function measure(x, y_sin, y_cos)
            s = Series(y_sin, y_cos)
            out = zeros(2)
            constant_interp!(out, x, s, 0.5)  # warmup
            return @allocated constant_interp!(out, x, s, 0.5)
        end
        @test measure(x, y_sin, y_cos) <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation (in-place vector)" begin
        function measure(x, y_sin, y_cos)
            s = Series(y_sin, y_cos)
            xqs = [0.1, 0.37, 0.5, 0.9]
            outputs = [zeros(length(xqs)) for _ in 1:2]
            constant_interp!(outputs, x, s, xqs)  # warmup
            return @allocated constant_interp!(outputs, x, s, xqs)
        end
        @test measure(x, y_sin, y_cos) <= ALLOC_THRESHOLD
    end

    @testset "Eltype contract: Integer series stays Int" begin
        x_f = collect(0.0:1.0:4.0)
        y1_int = [0, 1, 3, 4, 7]
        y2_int = [2, 3, 1, 0, 5]
        vals = constant_interp(x_f, Series(y1_int, y2_int), 1.5)
        @test vals isa Vector{Int}
        @test vals[1] == constant_interp(x_f, y1_int, 1.5)
    end

    @testset "Eltype contract: FillExtrap fill_value must be compatible with y eltype" begin
        x_f = collect(0.0:1.0:4.0)
        y_int = [0, 1, 3, 4, 7]
        # Int series + Int fill_value → output is Int.
        vals = constant_interp(x_f, Series(y_int), 5.0; extrap = FillExtrap(9))
        @test vals[1] == 9
        # Int series + Float fill_value → InexactError (raw eltype contract is strict).
        @test_throws InexactError constant_interp(x_f, Series(y_int), 5.0; extrap = FillExtrap(0.5))
    end

    @testset "Scalar: Vector-of-Vectors" begin
        vals = constant_interp(x, Series([y_sin, y_cos, y_exp]), 0.37)
        ref = constant_interp(x, Series(y_sin, y_cos, y_exp), 0.37)
        @test vals ≈ ref
    end

    @testset "DimensionMismatch on wrong output size" begin
        s = Series(y_sin, y_cos)
        out_wrong = zeros(5)
        @test_throws DimensionMismatch constant_interp!(out_wrong, x, s, 0.5)
    end

    @testset "PeriodicBC — scalar, :inclusive" begin
        s = Series(y_sin, y_cos)
        vals = constant_interp(x, s, 0.37; bc = PeriodicBC())
        vals_wrap = constant_interp(x, s, 0.37 + 1.0; bc = PeriodicBC())
        @test vals ≈ vals_wrap atol = 1.0e-12
    end

    @testset "PeriodicBC — scalar, :exclusive FVM" begin
        xc = [0.5, 1.5, 2.5]
        s = Series([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        vals_in = constant_interp(xc, s, 2.4; bc)
        @test vals_in[1] ≈ 30.0
        @test vals_in[2] ≈ 3.0
    end

    @testset "PeriodicBC — scalar in-place" begin
        s = Series(y_sin, y_cos)
        out = zeros(2)
        constant_interp!(out, x, s, 0.37; bc = PeriodicBC())
        @test out ≈ constant_interp(x, s, 0.37; bc = PeriodicBC())
    end

    @testset "PeriodicBC — vector in-place, :inclusive" begin
        s = Series(y_sin, y_cos)
        xqs = [0.1, 0.37, 0.9]
        outputs = [zeros(length(xqs)) for _ in 1:2]
        constant_interp!(outputs, x, s, xqs; bc = PeriodicBC())
        outputs_wrap = [zeros(length(xqs)) for _ in 1:2]
        constant_interp!(outputs_wrap, x, s, xqs .+ 1.0; bc = PeriodicBC())
        @test outputs[1] ≈ outputs_wrap[1] atol = 1.0e-12
        @test outputs[2] ≈ outputs_wrap[2] atol = 1.0e-12
    end

    @testset "PeriodicBC — vector in-place, :exclusive" begin
        xc = [0.5, 1.5, 2.5]
        s = Series([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        xqs = [0.6, 1.4, 2.4]
        outputs = [zeros(3), zeros(3)]
        constant_interp!(outputs, xc, s, xqs; bc)
        @test outputs[1] ≈ [10.0, 20.0, 30.0]   # NearestSide picks closest cell center
        @test outputs[2] ≈ [1.0, 2.0, 3.0]
    end

    @testset "PeriodicBC — vector allocating" begin
        s = Series(y_sin, y_cos)
        outs = constant_interp(x, s, [0.1, 0.37]; bc = PeriodicBC())
        @test length(outs) == 2 && length(outs[1]) == 2
    end

    @testset "PeriodicBC — :inclusive endpoint mismatch raises" begin
        s = Series(y_sin, y_exp)
        @test_throws ArgumentError constant_interp(x, s, 0.37; bc = PeriodicBC())
    end

    @testset "PeriodicBC — :exclusive period too small raises" begin
        xv = [0.0, 1.0, 2.0, 3.0]
        s = Series([0.0, 1.0, 2.0, 3.0])
        bc_bad = PeriodicBC(endpoint = :exclusive, period = 2.5)
        @test_throws ArgumentError constant_interp(xv, s, 1.5; bc = bc_bad)
    end
end
