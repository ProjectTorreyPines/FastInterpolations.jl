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
        @test vals isa NTuple{3, Float64}
        @test collect(vals) ≈ ref
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
        @test vals isa Tuple{Float64}
        ref = linear_interp(x, y_sin, 0.5)
        @test vals[1] ≈ ref
    end

    @testset "Extrapolation modes" begin
        xq_oob = 1.5

        # NoExtrap: should throw
        @test_throws DomainError linear_interp(x, Series(y_sin, y_cos), xq_oob)

        # ClampExtrap
        vals_clamp = linear_interp(x, Series(y_sin, y_cos), xq_oob; extrap=ClampExtrap())
        ref_clamp_sin = linear_interp(x, y_sin, xq_oob; extrap=ClampExtrap())
        ref_clamp_cos = linear_interp(x, y_cos, xq_oob; extrap=ClampExtrap())
        @test vals_clamp[1] ≈ ref_clamp_sin
        @test vals_clamp[2] ≈ ref_clamp_cos

        # ExtendExtrap
        vals_ext = linear_interp(x, Series(y_sin, y_cos), xq_oob; extrap=ExtendExtrap())
        ref_ext_sin = linear_interp(x, y_sin, xq_oob; extrap=ExtendExtrap())
        ref_ext_cos = linear_interp(x, y_cos, xq_oob; extrap=ExtendExtrap())
        @test vals_ext[1] ≈ ref_ext_sin
        @test vals_ext[2] ≈ ref_ext_cos

        # WrapExtrap
        vals_wrap = linear_interp(x, Series(y_sin, y_cos), xq_oob; extrap=WrapExtrap())
        ref_wrap_sin = linear_interp(x, y_sin, xq_oob; extrap=WrapExtrap())
        ref_wrap_cos = linear_interp(x, y_cos, xq_oob; extrap=WrapExtrap())
        @test vals_wrap[1] ≈ ref_wrap_sin
        @test vals_wrap[2] ≈ ref_wrap_cos
    end

    @testset "Derivative ops" begin
        xq = 0.37
        for d in 0:3
            op = DerivOp(d)
            vals = linear_interp(x, Series(y_sin, y_cos), xq; deriv=op)
            ref_sin = linear_interp(x, y_sin, xq; deriv=op)
            ref_cos = linear_interp(x, y_cos, xq; deriv=op)
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
        @test vals isa NTuple{2, Float32}
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

    @testset "Zero allocation (Tuple scalar)" begin
        s = Series(y_sin, y_cos)
        f_alloc() = begin
            linear_interp(x, s, 0.5)
            return @allocated linear_interp(x, s, 0.5)
        end
        @test f_alloc() == 0
    end

    @testset "Zero allocation (in-place scalar)" begin
        s = Series(y_sin, y_cos)
        out = zeros(2)
        f_alloc() = begin
            linear_interp!(out, x, s, 0.5)
            return @allocated linear_interp!(out, x, s, 0.5)
        end
        @test f_alloc() == 0
    end
end
