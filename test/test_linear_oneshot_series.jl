using Test
using FastInterpolations

@testset "Linear One-Shot Series" begin
    x = collect(range(0.0, 1.0, 101))
    y_sin = sin.(2π .* x)
    y_cos = cos.(2π .* x)
    y_exp = exp.(x)

    # Reference: pre-built SeriesInterpolant
    sitp = linear_interp(x, Series(y_sin, y_cos, y_exp))

    @testset "Scalar: Tuple → NTuple" begin
        xq = 0.37
        vals = linear_interp(x, Series(y_sin, y_cos, y_exp), xq)
        ref = sitp(xq)
        @test vals isa Vector{Float64}
        @test vals ≈ ref
    end

    @testset "Scalar: Matrix → Vector" begin
        Y = hcat(y_sin, y_cos, y_exp)
        xq = 0.37
        vals = linear_interp(x, Series(Y), xq)
        ref = sitp(xq)
        @test vals isa Vector{Float64}
        @test vals ≈ ref
    end

    @testset "Scalar: Vector-of-Vectors → Vector" begin
        xq = 0.37
        vals = linear_interp(x, Series([y_sin, y_cos, y_exp]), xq)
        ref = sitp(xq)
        @test vals isa Vector{Float64}
        @test vals ≈ ref
    end

    @testset "In-place scalar" begin
        out = zeros(3)
        xq = 0.37
        ret = linear_interp!(out, x, Series(y_sin, y_cos, y_exp), xq)
        ref = sitp(xq)
        @test ret === out
        @test out ≈ ref
    end

    @testset "Vector query" begin
        xqs = [0.1, 0.37, 0.5, 0.9]
        outs = [zeros(length(xqs)) for _ in 1:3]
        linear_interp!(outs, x, Series(y_sin, y_cos, y_exp), xqs)
        for j in eachindex(xqs)
            ref = sitp(xqs[j])
            for k in 1:3
                @test outs[k][j] ≈ ref[k]
            end
        end
    end

    @testset "Vector query allocating" begin
        xqs = [0.1, 0.37, 0.5, 0.9]
        outs = linear_interp(x, Series(y_sin, y_cos, y_exp), xqs)
        @test length(outs) == 3
        @test length(outs[1]) == 4
        for j in eachindex(xqs)
            ref = sitp(xqs[j])
            for k in 1:3
                @test outs[k][j] ≈ ref[k]
            end
        end
    end

    @testset "Single series" begin
        vals = linear_interp(x, Series(y_sin), 0.5)
        @test vals isa Vector{Float64}
        @test length(vals) == 1
        ref = linear_interp(x, y_sin, 0.5)
        @test vals[1] ≈ ref
    end

    @testset "Extrapolation modes" begin
        xq_oob = 1.5

        # NoExtrap: should throw
        @test_throws DomainError linear_interp(x, Series(y_sin, y_cos), xq_oob)

        # ClampExtrap
        vals_clamp = linear_interp(x, Series(y_sin, y_cos), xq_oob; extrap = ClampExtrap())
        ref_clamp_sin = linear_interp(x, y_sin, xq_oob; extrap = ClampExtrap())
        ref_clamp_cos = linear_interp(x, y_cos, xq_oob; extrap = ClampExtrap())
        @test vals_clamp[1] ≈ ref_clamp_sin
        @test vals_clamp[2] ≈ ref_clamp_cos

        # ExtendExtrap
        vals_ext = linear_interp(x, Series(y_sin, y_cos), xq_oob; extrap = ExtendExtrap())
        ref_ext_sin = linear_interp(x, y_sin, xq_oob; extrap = ExtendExtrap())
        ref_ext_cos = linear_interp(x, y_cos, xq_oob; extrap = ExtendExtrap())
        @test vals_ext[1] ≈ ref_ext_sin
        @test vals_ext[2] ≈ ref_ext_cos

        # WrapExtrap
        vals_wrap = linear_interp(x, Series(y_sin, y_cos), xq_oob; extrap = WrapExtrap())
        ref_wrap_sin = linear_interp(x, y_sin, xq_oob; extrap = WrapExtrap())
        ref_wrap_cos = linear_interp(x, y_cos, xq_oob; extrap = WrapExtrap())
        @test vals_wrap[1] ≈ ref_wrap_sin
        @test vals_wrap[2] ≈ ref_wrap_cos

        # FillExtrap
        vals_fill = linear_interp(x, Series(y_sin, y_cos), xq_oob; extrap = FillExtrap(999.0))
        @test vals_fill[1] ≈ 999.0
        @test vals_fill[2] ≈ 999.0
    end

    @testset "Derivative ops" begin
        xq = 0.37
        for d in 0:3
            op = DerivOp(d)
            vals = linear_interp(x, Series(y_sin, y_cos), xq; deriv = op)
            ref_sin = linear_interp(x, y_sin, xq; deriv = op)
            ref_cos = linear_interp(x, y_cos, xq; deriv = op)
            @test vals[1] ≈ ref_sin
            @test vals[2] ≈ ref_cos
        end
    end

    @testset "Integer grid (Real promotion)" begin
        x_int = collect(0:10)
        y1 = Float64.(x_int .^ 2)
        y2 = Float64.(x_int .^ 3)
        vals = linear_interp(x_int, Series(y1, y2), 5.5)
        ref1 = linear_interp(x_int, y1, 5.5)
        ref2 = linear_interp(x_int, y2, 5.5)
        @test vals[1] ≈ ref1
        @test vals[2] ≈ ref2
    end

    @testset "Float32" begin
        x32 = Float32.(x)
        y32_sin = Float32.(y_sin)
        y32_cos = Float32.(y_cos)
        vals = linear_interp(x32, Series(y32_sin, y32_cos), 0.5f0)
        @test vals isa Vector{Float32}
        ref_sin = linear_interp(x32, y32_sin, 0.5f0)
        ref_cos = linear_interp(x32, y32_cos, 0.5f0)
        @test vals[1] ≈ ref_sin
        @test vals[2] ≈ ref_cos
    end

    @testset "Complex values" begin
        y_c1 = complex.(y_sin, y_cos)
        y_c2 = complex.(y_exp, -y_exp)
        vals = linear_interp(x, Series(y_c1, y_c2), 0.5)
        ref1 = linear_interp(x, y_c1, 0.5)
        ref2 = linear_interp(x, y_c2, 0.5)
        @test vals[1] ≈ ref1
        @test vals[2] ≈ ref2
    end

    @testset "Many series (20)" begin
        ys = [sin.(k .* 2π .* x) for k in 1:20]
        Y = hcat(ys...)
        vals = linear_interp(x, Series(Y), 0.5)
        @test length(vals) == 20
        for k in 1:20
            ref = linear_interp(x, ys[k], 0.5)
            @test vals[k] ≈ ref
        end
    end

    @testset "Range grid (O(1) search)" begin
        xr = range(0.0, 1.0, 101)
        yr_sin = sin.(2π .* xr)
        yr_cos = cos.(2π .* xr)
        vals = linear_interp(xr, Series(collect(yr_sin), collect(yr_cos)), 0.37)
        ref_sin = linear_interp(xr, collect(yr_sin), 0.37)
        ref_cos = linear_interp(xr, collect(yr_cos), 0.37)
        @test vals[1] ≈ ref_sin
        @test vals[2] ≈ ref_cos
    end

    @testset "Zero allocation (in-place scalar)" begin
        s = Series(y_sin, y_cos)
        out = zeros(2)
        f_alloc() = begin
            linear_interp!(out, x, s, 0.5)
            return @allocated linear_interp!(out, x, s, 0.5)
        end
        @test f_alloc() <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation (in-place vector)" begin
        s = Series(y_sin, y_cos)
        xqs = [0.1, 0.37, 0.5, 0.9]
        outputs = [zeros(length(xqs)) for _ in 1:2]
        f_alloc() = begin
            linear_interp!(outputs, x, s, xqs)
            return @allocated linear_interp!(outputs, x, s, xqs)
        end
        @test f_alloc() <= ALLOC_THRESHOLD
    end

    @testset "ForwardDiff AD" begin
        using ForwardDiff
        # AD derivative should match DerivOp(1) result (analytic slope)
        f_ad(t) = sum(linear_interp(x, Series(y_sin, y_cos), t))
        grad = ForwardDiff.derivative(f_ad, 0.37)
        d1 = linear_interp(x, Series(y_sin, y_cos), 0.37; deriv = DerivOp(1))
        @test grad ≈ sum(d1)
    end

    @testset "Type promotion: Integer series" begin
        x_f = collect(0.0:1.0:4.0)
        y1_int = [0, 1, 3, 4, 7]
        y2_int = [2, 3, 1, 0, 5]
        vals = linear_interp(x_f, Series(y1_int, y2_int), 1.5)
        @test vals isa Vector{Float64}
        ref1 = linear_interp(x_f, Float64.(y1_int), 1.5)
        @test vals[1] ≈ ref1
    end

    @testset "Type promotion: Mixed Float32/Float64 series" begin
        y32 = Float32.(y_sin)
        y64 = y_cos  # Float64
        vals = linear_interp(x, Series(y32, y64), 0.37)
        @test vals isa Vector{Float64}
    end

    @testset "Type promotion: FillExtrap with Integer series" begin
        x_f = collect(0.0:1.0:4.0)
        y_int = [0, 1, 3, 4, 7]
        vals = linear_interp(x_f, Series(y_int), 5.0; extrap = FillExtrap(0.5))
        @test vals[1] ≈ 0.5
    end

    @testset "DimensionMismatch on wrong output size" begin
        s = Series(y_sin, y_cos)
        out_wrong = zeros(5)
        @test_throws DimensionMismatch linear_interp!(out_wrong, x, s, 0.5)
    end

    @testset "Mixed-precision vector query (Float32 queries on Float64 grid)" begin
        s = Series(y_sin, y_cos)
        xqs_f32 = Float32[0.1, 0.37, 0.5, 0.9]
        outputs = [zeros(length(xqs_f32)) for _ in 1:2]
        # Should not throw — pool buffer type must accommodate promoted Tq
        @test begin
            linear_interp!(outputs, x, s, xqs_f32)
            true
        end
        # Results should match Float64 queries
        xqs_f64 = Float64.(xqs_f32)
        outputs_ref = [zeros(length(xqs_f64)) for _ in 1:2]
        linear_interp!(outputs_ref, x, s, xqs_f64)
        @test outputs[1] ≈ outputs_ref[1]
        @test outputs[2] ≈ outputs_ref[2]
    end

    @testset "PeriodicBC — scalar, :inclusive" begin
        s = Series(y_sin, y_cos)   # y[1] == y[end] = 0
        vals = linear_interp(x, s, 0.37; bc = PeriodicBC())
        vals_wrap = linear_interp(x, s, 0.37 + 1.0; bc = PeriodicBC())
        @test vals ≈ vals_wrap atol = 1.0e-12
    end

    @testset "PeriodicBC — scalar, :exclusive FVM" begin
        xc = [0.5, 1.5, 2.5]
        s = Series([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        @test linear_interp(xc, s, 3.0; bc)[1] ≈ 20.0 atol = 1.0e-12   # midpoint of extended [2.5, 3.5]
        @test linear_interp(xc, s, 3.0; bc)[2] ≈ 2.0 atol = 1.0e-12
    end

    @testset "PeriodicBC — scalar in-place" begin
        s = Series(y_sin, y_cos)
        out = zeros(2)
        linear_interp!(out, x, s, 0.37; bc = PeriodicBC())
        @test out ≈ linear_interp(x, s, 0.37; bc = PeriodicBC())
    end

    @testset "PeriodicBC — vector in-place, :inclusive" begin
        s = Series(y_sin, y_cos)
        xqs = [0.1, 0.37, 0.9]
        outputs = [zeros(length(xqs)) for _ in 1:2]
        linear_interp!(outputs, x, s, xqs; bc = PeriodicBC())
        outputs_wrap = [zeros(length(xqs)) for _ in 1:2]
        linear_interp!(outputs_wrap, x, s, xqs .+ 1.0; bc = PeriodicBC())
        @test outputs[1] ≈ outputs_wrap[1] atol = 1.0e-12
        @test outputs[2] ≈ outputs_wrap[2] atol = 1.0e-12
    end

    @testset "PeriodicBC — vector in-place, :exclusive" begin
        xc = [0.5, 1.5, 2.5]
        s = Series([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        xqs = [0.5, 2.5, 3.0]
        outputs = [zeros(3), zeros(3)]
        linear_interp!(outputs, xc, s, xqs; bc)
        @test outputs[1][2] ≈ 30.0 atol = 1.0e-12
        @test outputs[1][3] ≈ 20.0 atol = 1.0e-12   # midpoint
        @test outputs[2][3] ≈ 2.0 atol = 1.0e-12
    end

    @testset "PeriodicBC — vector allocating" begin
        s = Series(y_sin, y_cos)
        outs = linear_interp(x, s, [0.1, 0.37]; bc = PeriodicBC())
        @test length(outs) == 2 && length(outs[1]) == 2
    end

    @testset "PeriodicBC — :inclusive endpoint mismatch raises" begin
        s = Series(y_sin, y_exp)   # y_exp[1] != y_exp[end]
        @test_throws ArgumentError linear_interp(x, s, 0.37; bc = PeriodicBC())
    end

    @testset "PeriodicBC — :exclusive period too small raises" begin
        # Vector grid span = 3.0, period = 2.5 → virtual endpoint < last(x) → throw
        xv = [0.0, 1.0, 2.0, 3.0]
        s = Series([0.0, 1.0, 2.0, 3.0])
        bc_bad = PeriodicBC(endpoint = :exclusive, period = 2.5)
        @test_throws ArgumentError linear_interp(xv, s, 1.5; bc = bc_bad)
    end
end
