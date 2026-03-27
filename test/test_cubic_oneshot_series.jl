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
        @test vals isa NTuple{3, Float64}
        @test collect(vals) ≈ ref
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
end
