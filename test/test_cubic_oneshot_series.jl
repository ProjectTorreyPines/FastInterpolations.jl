using Test
using FastInterpolations

@testset "Cubic One-Shot Series" begin
    x = collect(range(0.0, 1.0, 101))
    y_sin = sin.(2π .* x)
    y_cos = cos.(2π .* x)
    y_exp = exp.(x)

    sitp = cubic_interp(x, Series(y_sin, y_cos, y_exp))

    @testset "Scalar: Tuple → NTuple (CubicFit)" begin
        xq = 0.37
        vals = cubic_interp(x, Series(y_sin, y_cos, y_exp), xq)
        ref = sitp(xq)
        @test vals isa Vector{Float64}
        @test vals ≈ ref
    end

    @testset "Scalar: Matrix → Vector" begin
        Y = hcat(y_sin, y_cos, y_exp)
        vals = cubic_interp(x, Series(Y), 0.37)
        ref = sitp(0.37)
        @test vals isa Vector{Float64}
        @test vals ≈ ref
    end

    @testset "In-place scalar" begin
        out = zeros(3)
        ret = cubic_interp!(out, x, Series(y_sin, y_cos, y_exp), 0.37)
        ref = sitp(0.37)
        @test ret === out
        @test out ≈ ref
    end

    @testset "Vector query" begin
        xqs = [0.1, 0.37, 0.5, 0.9]
        outs = cubic_interp(x, Series(y_sin, y_cos, y_exp), xqs)
        @test length(outs) == 3
        for j in eachindex(xqs)
            ref = sitp(xqs[j])
            for k in 1:3
                @test outs[k][j] ≈ ref[k]
            end
        end
    end

    @testset "BC types" begin
        for bc_val in [CubicFit(), FastInterpolations.Deriv2(0.0)]
            vals = cubic_interp(x, Series(y_sin, y_cos), 0.37; bc=bc_val)
            ref_sin = cubic_interp(x, y_sin, 0.37; bc=bc_val)
            ref_cos = cubic_interp(x, y_cos, 0.37; bc=bc_val)
            @test vals[1] ≈ ref_sin
            @test vals[2] ≈ ref_cos
        end
    end

    @testset "PeriodicBC" begin
        vals = cubic_interp(x, Series(y_sin, y_cos), 0.37; bc=PeriodicBC())
        ref_sin = cubic_interp(x, y_sin, 0.37; bc=PeriodicBC())
        ref_cos = cubic_interp(x, y_cos, 0.37; bc=PeriodicBC())
        @test vals[1] ≈ ref_sin
        @test vals[2] ≈ ref_cos
    end

    @testset "Zero allocation (in-place scalar, CubicFit)" begin
        s = Series(y_sin, y_cos)
        out = zeros(2)
        f_alloc() = begin
            cubic_interp!(out, x, s, 0.37)
            return @allocated cubic_interp!(out, x, s, 0.37)
        end
        @test f_alloc() <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation (in-place scalar, PeriodicBC)" begin
        s = Series(y_sin, y_cos)
        out = zeros(2)
        f_alloc() = begin
            cubic_interp!(out, x, s, 0.37; bc = PeriodicBC())
            return @allocated cubic_interp!(out, x, s, 0.37; bc = PeriodicBC())
        end
        @test f_alloc() <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation (in-place vector, CubicFit)" begin
        s = Series(y_sin, y_cos)
        xqs = [0.1, 0.37, 0.5, 0.9]
        outputs = [zeros(length(xqs)) for _ in 1:2]
        f_alloc() = begin
            cubic_interp!(outputs, x, s, xqs)
            return @allocated cubic_interp!(outputs, x, s, xqs)
        end
        # Vector path has small container allocation for z-buffer array
        # (pool buffers reused, but [acquire!(...) for _] allocates the Vector container)
        @test f_alloc() <= 240
    end

    @testset "Extrapolation modes" begin
        xq_oob = 1.5
        @test_throws DomainError cubic_interp(x, Series(y_sin, y_cos), xq_oob)

        vals_clamp = cubic_interp(x, Series(y_sin, y_cos), xq_oob; extrap=ClampExtrap())
        ref_sin = cubic_interp(x, y_sin, xq_oob; extrap=ClampExtrap())
        ref_cos = cubic_interp(x, y_cos, xq_oob; extrap=ClampExtrap())
        @test vals_clamp[1] ≈ ref_sin
        @test vals_clamp[2] ≈ ref_cos

        vals_ext = cubic_interp(x, Series(y_sin, y_cos), xq_oob; extrap=ExtendExtrap())
        ref_sin_ext = cubic_interp(x, y_sin, xq_oob; extrap=ExtendExtrap())
        ref_cos_ext = cubic_interp(x, y_cos, xq_oob; extrap=ExtendExtrap())
        @test vals_ext[1] ≈ ref_sin_ext
        @test vals_ext[2] ≈ ref_cos_ext

        # WrapExtrap
        vals_wrap = cubic_interp(x, Series(y_sin, y_cos), xq_oob; extrap=WrapExtrap())
        ref_wrap_sin = cubic_interp(x, y_sin, xq_oob; extrap=WrapExtrap())
        ref_wrap_cos = cubic_interp(x, y_cos, xq_oob; extrap=WrapExtrap())
        @test vals_wrap[1] ≈ ref_wrap_sin
        @test vals_wrap[2] ≈ ref_wrap_cos

        # FillExtrap
        vals_fill = cubic_interp(x, Series(y_sin, y_cos), xq_oob; extrap=FillExtrap(999.0))
        @test vals_fill[1] ≈ 999.0
        @test vals_fill[2] ≈ 999.0
    end

    @testset "Derivative ops" begin
        for d in 0:4
            op = DerivOp(d)
            vals = cubic_interp(x, Series(y_sin, y_cos), 0.37; deriv=op)
            ref_sin = cubic_interp(x, y_sin, 0.37; deriv=op)
            ref_cos = cubic_interp(x, y_cos, 0.37; deriv=op)
            @test vals[1] ≈ ref_sin
            @test vals[2] ≈ ref_cos
        end
    end

    @testset "Integer grid (Real promotion)" begin
        x_int = collect(0:10)
        y1 = Float64.(x_int .^ 2)
        y2 = Float64.(x_int .^ 3)
        vals = cubic_interp(x_int, Series(y1, y2), 5.5)
        ref1 = cubic_interp(x_int, y1, 5.5)
        ref2 = cubic_interp(x_int, y2, 5.5)
        @test vals[1] ≈ ref1
        @test vals[2] ≈ ref2
    end

    @testset "Many series (20)" begin
        ys = [sin.(k .* 2π .* x) for k in 1:20]
        Y = hcat(ys...)
        vals = cubic_interp(x, Series(Y), 0.5)
        @test length(vals) == 20
        for k in 1:20
            ref = cubic_interp(x, ys[k], 0.5)
            @test vals[k] ≈ ref
        end
    end

    @testset "Single series" begin
        vals = cubic_interp(x, Series(y_sin), 0.5)
        ref = cubic_interp(x, y_sin, 0.5)
        @test vals[1] ≈ ref
    end

    @testset "Float32" begin
        x32 = Float32.(x)
        y32_sin = Float32.(y_sin)
        y32_cos = Float32.(y_cos)
        vals = cubic_interp(x32, Series(y32_sin, y32_cos), 0.37f0)
        @test vals isa Vector{Float32}
        ref_sin = cubic_interp(x32, y32_sin, 0.37f0)
        ref_cos = cubic_interp(x32, y32_cos, 0.37f0)
        @test vals[1] ≈ ref_sin
        @test vals[2] ≈ ref_cos
    end

    @testset "Complex values" begin
        y_c1 = complex.(y_sin, y_cos)
        y_c2 = complex.(y_exp, -y_exp)
        vals = cubic_interp(x, Series(y_c1, y_c2), 0.5)
        ref1 = cubic_interp(x, y_c1, 0.5)
        ref2 = cubic_interp(x, y_c2, 0.5)
        @test vals[1] ≈ ref1
        @test vals[2] ≈ ref2
    end

    @testset "ForwardDiff AD" begin
        using ForwardDiff
        f_ad(t) = sum(cubic_interp(x, Series(y_sin, y_cos), t))
        grad = ForwardDiff.derivative(f_ad, 0.37)
        d1 = cubic_interp(x, Series(y_sin, y_cos), 0.37; deriv=DerivOp(1))
        @test grad ≈ sum(d1)
    end

    @testset "Type promotion: Integer series" begin
        x_f = collect(0.0:1.0:4.0)
        y1_int = [0, 1, 3, 4, 7]
        y2_int = [2, 3, 1, 0, 5]
        vals = cubic_interp(x_f, Series(y1_int, y2_int), 1.5)
        @test vals isa Vector{Float64}
        ref1 = cubic_interp(x_f, Float64.(y1_int), 1.5)
        @test vals[1] ≈ ref1
    end

    @testset "Type promotion: Mixed Float32/Float64 series" begin
        y32 = Float32.(y_sin)
        y64 = y_cos
        vals = cubic_interp(x, Series(y32, y64), 0.37)
        @test vals isa Vector{Float64}
    end
end
