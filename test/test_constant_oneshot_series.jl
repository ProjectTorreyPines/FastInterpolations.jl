using Test
using FastInterpolations

@testset "Constant One-Shot Series" begin
    x = collect(range(0.0, 1.0, 101))
    y_sin = sin.(2π .* x)
    y_cos = cos.(2π .* x)
    y_exp = exp.(x)

    sitp = constant_interp(x, Series(y_sin, y_cos, y_exp))

    @testset "Scalar: Tuple → NTuple" begin
        xq = 0.37
        vals = constant_interp(x, Series(y_sin, y_cos, y_exp), xq)
        ref = sitp(xq)
        @test vals isa NTuple{3, Float64}
        @test collect(vals) ≈ ref
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
            sitp_s = constant_interp(x, Series(y_sin, y_cos); side=side_opt)
            vals = constant_interp(x, Series(y_sin, y_cos), 0.37; side=side_opt)
            ref = sitp_s(0.37)
            @test collect(vals) ≈ ref
        end
    end

    @testset "Extrapolation modes" begin
        xq_oob = 1.5
        @test_throws DomainError constant_interp(x, Series(y_sin, y_cos), xq_oob)

        vals_clamp = constant_interp(x, Series(y_sin, y_cos), xq_oob; extrap=ClampExtrap())
        ref_sin = constant_interp(x, y_sin, xq_oob; extrap=ClampExtrap())
        ref_cos = constant_interp(x, y_cos, xq_oob; extrap=ClampExtrap())
        @test vals_clamp[1] ≈ ref_sin
        @test vals_clamp[2] ≈ ref_cos

        vals_ext = constant_interp(x, Series(y_sin, y_cos), xq_oob; extrap=ExtendExtrap())
        ref_sin_ext = constant_interp(x, y_sin, xq_oob; extrap=ExtendExtrap())
        @test vals_ext[1] ≈ ref_sin_ext
    end

    @testset "Derivative ops" begin
        for d in 0:2
            op = DerivOp(d)
            vals = constant_interp(x, Series(y_sin, y_cos), 0.37; deriv=op)
            ref_sin = constant_interp(x, y_sin, 0.37; deriv=op)
            ref_cos = constant_interp(x, y_cos, 0.37; deriv=op)
            @test vals[1] ≈ ref_sin
            @test vals[2] ≈ ref_cos
        end
    end

    @testset "Zero allocation (Tuple scalar)" begin
        s = Series(y_sin, y_cos)
        f_alloc() = begin
            constant_interp(x, s, 0.5)
            return @allocated constant_interp(x, s, 0.5)
        end
        @test f_alloc() == 0
    end
end
