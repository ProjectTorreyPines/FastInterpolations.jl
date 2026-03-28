using Test
using FastInterpolations

@testset "Quadratic One-Shot Series" begin
    x = collect(range(0.0, 1.0, 101))
    y_sin = sin.(2π .* x)
    y_cos = cos.(2π .* x)
    y_exp = exp.(x)

    @testset "Scalar: Tuple → NTuple" begin
        vals = quadratic_interp(x, Series(y_sin, y_cos, y_exp), 0.37)
        ref_sin = quadratic_interp(x, y_sin, 0.37)
        ref_cos = quadratic_interp(x, y_cos, 0.37)
        ref_exp = quadratic_interp(x, y_exp, 0.37)
        @test vals isa Vector{Float64}
        @test vals[1] ≈ ref_sin
        @test vals[2] ≈ ref_cos
        @test vals[3] ≈ ref_exp
    end

    @testset "Scalar: Matrix → Vector" begin
        Y = hcat(y_sin, y_cos, y_exp)
        vals = quadratic_interp(x, Series(Y), 0.37)
        ref_sin = quadratic_interp(x, y_sin, 0.37)
        @test vals isa Vector{Float64}
        @test vals[1] ≈ ref_sin
    end

    @testset "In-place scalar" begin
        out = zeros(3)
        ret = quadratic_interp!(out, x, Series(y_sin, y_cos, y_exp), 0.37)
        ref_sin = quadratic_interp(x, y_sin, 0.37)
        @test ret === out
        @test out[1] ≈ ref_sin
    end

    @testset "Vector query" begin
        xqs = [0.1, 0.37, 0.5, 0.9]
        outs = quadratic_interp(x, Series(y_sin, y_cos), xqs)
        @test length(outs) == 2
        for j in eachindex(xqs)
            ref_sin = quadratic_interp(x, y_sin, xqs[j])
            ref_cos = quadratic_interp(x, y_cos, xqs[j])
            @test outs[1][j] ≈ ref_sin
            @test outs[2][j] ≈ ref_cos
        end
    end

    @testset "BC types" begin
        for bc_val in [Left(QuadraticFit()), Right(QuadraticFit())]
            vals = quadratic_interp(x, Series(y_sin, y_cos), 0.37; bc=bc_val)
            ref_sin = quadratic_interp(x, y_sin, 0.37; bc=bc_val)
            ref_cos = quadratic_interp(x, y_cos, 0.37; bc=bc_val)
            @test vals[1] ≈ ref_sin
            @test vals[2] ≈ ref_cos
        end
    end

    @testset "Extrapolation modes" begin
        xq_oob = 1.5
        @test_throws DomainError quadratic_interp(x, Series(y_sin, y_cos), xq_oob)

        vals_clamp = quadratic_interp(x, Series(y_sin, y_cos), xq_oob; extrap=ClampExtrap())
        ref_sin = quadratic_interp(x, y_sin, xq_oob; extrap=ClampExtrap())
        @test vals_clamp[1] ≈ ref_sin

        vals_ext = quadratic_interp(x, Series(y_sin, y_cos), xq_oob; extrap=ExtendExtrap())
        ref_sin_ext = quadratic_interp(x, y_sin, xq_oob; extrap=ExtendExtrap())
        @test vals_ext[1] ≈ ref_sin_ext

        # WrapExtrap
        vals_wrap = quadratic_interp(x, Series(y_sin, y_cos), xq_oob; extrap=WrapExtrap())
        ref_wrap_sin = quadratic_interp(x, y_sin, xq_oob; extrap=WrapExtrap())
        ref_wrap_cos = quadratic_interp(x, y_cos, xq_oob; extrap=WrapExtrap())
        @test vals_wrap[1] ≈ ref_wrap_sin
        @test vals_wrap[2] ≈ ref_wrap_cos

        # FillExtrap
        vals_fill = quadratic_interp(x, Series(y_sin, y_cos), xq_oob; extrap=FillExtrap(999.0))
        @test vals_fill[1] ≈ 999.0
        @test vals_fill[2] ≈ 999.0
    end

    @testset "Derivative ops" begin
        for d in 0:3
            op = DerivOp(d)
            vals = quadratic_interp(x, Series(y_sin, y_cos), 0.37; deriv=op)
            ref_sin = quadratic_interp(x, y_sin, 0.37; deriv=op)
            ref_cos = quadratic_interp(x, y_cos, 0.37; deriv=op)
            @test vals[1] ≈ ref_sin
            @test vals[2] ≈ ref_cos
        end
    end

    @testset "Integer grid (Real promotion)" begin
        x_int = collect(0:10)
        y1 = Float64.(x_int .^ 2)
        y2 = Float64.(x_int .^ 3)
        vals = quadratic_interp(x_int, Series(y1, y2), 5.5)
        ref1 = quadratic_interp(x_int, y1, 5.5)
        ref2 = quadratic_interp(x_int, y2, 5.5)
        @test vals[1] ≈ ref1
        @test vals[2] ≈ ref2
    end

    @testset "Many series (20)" begin
        ys = [sin.(k .* 2π .* x) for k in 1:20]
        Y = hcat(ys...)
        vals = quadratic_interp(x, Series(Y), 0.5)
        @test length(vals) == 20
        for k in 1:20
            ref = quadratic_interp(x, ys[k], 0.5)
            @test vals[k] ≈ ref
        end
    end

    @testset "Float32" begin
        x32 = Float32.(x)
        y32_sin = Float32.(y_sin)
        y32_cos = Float32.(y_cos)
        vals = quadratic_interp(x32, Series(y32_sin, y32_cos), 0.5f0)
        @test vals isa Vector{Float32}
        ref_sin = quadratic_interp(x32, y32_sin, 0.5f0)
        @test vals[1] ≈ ref_sin
    end

    @testset "Complex values" begin
        y_c1 = complex.(y_sin, y_cos)
        y_c2 = complex.(y_exp, -y_exp)
        vals = quadratic_interp(x, Series(y_c1, y_c2), 0.5)
        ref1 = quadratic_interp(x, y_c1, 0.5)
        ref2 = quadratic_interp(x, y_c2, 0.5)
        @test vals[1] ≈ ref1
        @test vals[2] ≈ ref2
    end

    @testset "ForwardDiff AD" begin
        using ForwardDiff
        f_ad(t) = sum(quadratic_interp(x, Series(y_sin, y_cos), t))
        grad = ForwardDiff.derivative(f_ad, 0.37)
        d1 = quadratic_interp(x, Series(y_sin, y_cos), 0.37; deriv=DerivOp(1))
        @test grad ≈ sum(d1)
    end

    @testset "Zero allocation (in-place scalar)" begin
        s = Series(y_sin, y_cos)
        out = zeros(2)
        f_alloc() = begin
            quadratic_interp!(out, x, s, 0.5)
            return @allocated quadratic_interp!(out, x, s, 0.5)
        end
        @test f_alloc() <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation (in-place vector)" begin
        s = Series(y_sin, y_cos)
        xqs = [0.1, 0.37, 0.5, 0.9]
        outputs = [zeros(length(xqs)) for _ in 1:2]
        f_alloc() = begin
            quadratic_interp!(outputs, x, s, xqs)
            return @allocated quadratic_interp!(outputs, x, s, xqs)
        end
        # Vector path has small container allocation for coefficient arrays
        # (pool buffers reused, but [acquire!(...) for _] allocates the Vector container)
        @test f_alloc() <= 240
    end

    @testset "Type promotion: Integer series" begin
        x_f = collect(0.0:1.0:4.0)
        y1_int = [0, 1, 3, 4, 7]
        y2_int = [2, 3, 1, 0, 5]
        vals = quadratic_interp(x_f, Series(y1_int, y2_int), 1.5)
        @test vals isa Vector{Float64}
        ref1 = quadratic_interp(x_f, Float64.(y1_int), 1.5)
        @test vals[1] ≈ ref1
    end

    @testset "Type promotion: Mixed Float32/Float64 series" begin
        y32 = Float32.(y_sin)
        y64 = y_cos
        vals = quadratic_interp(x, Series(y32, y64), 0.37)
        @test vals isa Vector{Float64}
    end
end
