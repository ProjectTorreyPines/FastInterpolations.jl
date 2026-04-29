@testitem "Cubic One-Shot Series" setup = [AllocConstants] begin
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
            vals = cubic_interp(x, Series(y_sin, y_cos), 0.37; bc = bc_val)
            ref_sin = cubic_interp(x, y_sin, 0.37; bc = bc_val)
            ref_cos = cubic_interp(x, y_cos, 0.37; bc = bc_val)
            @test vals[1] ≈ ref_sin
            @test vals[2] ≈ ref_cos
        end
    end

    @testset "PeriodicBC scalar" begin
        vals = cubic_interp(x, Series(y_sin, y_cos), 0.37; bc = PeriodicBC())
        ref_sin = cubic_interp(x, y_sin, 0.37; bc = PeriodicBC())
        ref_cos = cubic_interp(x, y_cos, 0.37; bc = PeriodicBC())
        @test vals[1] ≈ ref_sin
        @test vals[2] ≈ ref_cos
    end

    @testset "PeriodicBC vector (inclusive)" begin
        xqs = [0.1, 0.37, 0.5, 0.9]
        outs = cubic_interp(x, Series(y_sin, y_cos), xqs; bc = PeriodicBC())
        @test length(outs) == 2
        for j in eachindex(xqs)
            ref_sin = cubic_interp(x, y_sin, xqs[j]; bc = PeriodicBC())
            ref_cos = cubic_interp(x, y_cos, xqs[j]; bc = PeriodicBC())
            @test outs[1][j] ≈ ref_sin
            @test outs[2][j] ≈ ref_cos
        end
    end

    @testset "PeriodicBC vector (exclusive)" begin
        # Exclusive: last point != first point, period = 1.0
        x_exc = collect(range(0.0, step = 0.01, length = 100))  # does NOT include 1.0
        y1_exc = sin.(2π .* x_exc)
        y2_exc = cos.(2π .* x_exc)
        bc_exc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        xqs = [0.05, 0.37, 0.75, 0.95]
        outs = cubic_interp(x_exc, Series(y1_exc, y2_exc), xqs; bc = bc_exc)
        @test length(outs) == 2
        for j in eachindex(xqs)
            ref1 = cubic_interp(x_exc, y1_exc, xqs[j]; bc = bc_exc)
            ref2 = cubic_interp(x_exc, y2_exc, xqs[j]; bc = bc_exc)
            @test outs[1][j] ≈ ref1
            @test outs[2][j] ≈ ref2
        end
    end

    # Pin the seam region for both scalar and vector Series oneshot under
    # zero-copy exclusive periodic. Reference is the single-vector 1D path,
    # which routes the seam pair `(n, 1)` through `search_interval`.
    @testset "PeriodicBC scalar (exclusive) — seam region" begin
        x_exc = collect(range(0.0, step = 0.01, length = 100))
        y1_exc = sin.(2π .* x_exc)
        y2_exc = cos.(2π .* x_exc)
        bc_exc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        for xq in (0.991, 0.995, 0.999)
            vals = cubic_interp(x_exc, Series(y1_exc, y2_exc), xq; bc = bc_exc)
            ref1 = cubic_interp(x_exc, y1_exc, xq; bc = bc_exc)
            ref2 = cubic_interp(x_exc, y2_exc, xq; bc = bc_exc)
            @test vals[1] ≈ ref1 atol = 1e-12
            @test vals[2] ≈ ref2 atol = 1e-12
        end
    end

    # Float32 grid + Float64 period: `WrapExtrap(x, bc)` must cast the period to
    # the grid's float type, otherwise OOB-wrapped queries widen to Float64 and
    # break the preallocated `_CubicAnchoredQuery{Float32, Float32}` buffer.
    @testset "Float32 grid + Float64 period vector series" begin
        x32 = collect(range(0.0f0, step = 0.1f0, length = 10))   # Float32, span 0.9
        # Keep y in Float32 (default `2π` is Float64 → broadcast widens).
        two_pi = Float32(2π)
        y1 = sin.(two_pi .* x32)
        y2 = cos.(two_pi .* x32)
        @assert eltype(y1) === Float32
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)     # Float64 literal
        # 1.05 is OOB, wraps via the seam-aware WrapExtrap.
        xqs = Float32[0.05, 1.05]
        outs = cubic_interp(x32, Series(y1, y2), xqs; bc = bc)
        @test length(outs) == 2
        @test eltype(outs[1]) === Float32
        for j in eachindex(xqs)
            ref1 = cubic_interp(x32, y1, xqs[j]; bc = bc)
            ref2 = cubic_interp(x32, y2, xqs[j]; bc = bc)
            @test outs[1][j] ≈ ref1
            @test outs[2][j] ≈ ref2
        end
    end

    @testset "PeriodicBC vector (exclusive) — seam region" begin
        x_exc = collect(range(0.0, step = 0.01, length = 100))
        y1_exc = sin.(2π .* x_exc)
        y2_exc = cos.(2π .* x_exc)
        bc_exc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        xqs_seam = [0.991, 0.995, 0.999]  # all in [x[n], x[1]+period) seam cell
        outs = cubic_interp(x_exc, Series(y1_exc, y2_exc), xqs_seam; bc = bc_exc)
        for j in eachindex(xqs_seam)
            ref1 = cubic_interp(x_exc, y1_exc, xqs_seam[j]; bc = bc_exc)
            ref2 = cubic_interp(x_exc, y2_exc, xqs_seam[j]; bc = bc_exc)
            @test outs[1][j] ≈ ref1 atol = 1e-12
            @test outs[2][j] ≈ ref2 atol = 1e-12
        end
    end

    @testset "PeriodicBC vector in-place" begin
        xqs = [0.1, 0.37, 0.5, 0.9]
        outputs = [zeros(length(xqs)) for _ in 1:2]
        ret = cubic_interp!(outputs, x, Series(y_sin, y_cos), xqs; bc = PeriodicBC())
        @test ret === outputs
        outs_alloc = cubic_interp(x, Series(y_sin, y_cos), xqs; bc = PeriodicBC())
        for k in 1:2
            @test outputs[k] ≈ outs_alloc[k]
        end
    end

    # NOTE: Allocation tests use the function-barrier pattern with arguments
    # (rather than closure capture or @testset-local vars) because Test.jl wraps
    # @testset bodies in try/catch, which weakens type inference under @testitem
    # fresh-module compilation. Passing outer vars as args restores type stability.
    @testset "Zero allocation (in-place vector, PeriodicBC)" begin
        function measure(x, y_sin, y_cos)
            s = Series(y_sin, y_cos)
            xqs = [0.1, 0.37, 0.5, 0.9]
            outputs = [zeros(length(xqs)) for _ in 1:2]
            cubic_interp!(outputs, x, s, xqs; bc = PeriodicBC())  # warmup
            return @allocated cubic_interp!(outputs, x, s, xqs; bc = PeriodicBC())
        end
        @test measure(x, y_sin, y_cos) <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation (in-place scalar, CubicFit)" begin
        function measure(x, y_sin, y_cos)
            s = Series(y_sin, y_cos)
            out = zeros(2)
            cubic_interp!(out, x, s, 0.37)  # warmup
            return @allocated cubic_interp!(out, x, s, 0.37)
        end
        @test measure(x, y_sin, y_cos) <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation (in-place scalar, PeriodicBC)" begin
        function measure(x, y_sin, y_cos)
            s = Series(y_sin, y_cos)
            out = zeros(2)
            cubic_interp!(out, x, s, 0.37; bc = PeriodicBC())  # warmup
            return @allocated cubic_interp!(out, x, s, 0.37; bc = PeriodicBC())
        end
        @test measure(x, y_sin, y_cos) <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation (in-place vector, CubicFit)" begin
        function measure(x, y_sin, y_cos)
            s = Series(y_sin, y_cos)
            xqs = [0.1, 0.37, 0.5, 0.9]
            outputs = [zeros(length(xqs)) for _ in 1:2]
            cubic_interp!(outputs, x, s, xqs)  # warmup
            return @allocated cubic_interp!(outputs, x, s, xqs)
        end
        @test measure(x, y_sin, y_cos) <= ALLOC_THRESHOLD
    end

    @testset "Extrapolation modes" begin
        xq_oob = 1.5
        @test_throws DomainError cubic_interp(x, Series(y_sin, y_cos), xq_oob)

        vals_clamp = cubic_interp(x, Series(y_sin, y_cos), xq_oob; extrap = ClampExtrap())
        ref_sin = cubic_interp(x, y_sin, xq_oob; extrap = ClampExtrap())
        ref_cos = cubic_interp(x, y_cos, xq_oob; extrap = ClampExtrap())
        @test vals_clamp[1] ≈ ref_sin
        @test vals_clamp[2] ≈ ref_cos

        vals_ext = cubic_interp(x, Series(y_sin, y_cos), xq_oob; extrap = ExtendExtrap())
        ref_sin_ext = cubic_interp(x, y_sin, xq_oob; extrap = ExtendExtrap())
        ref_cos_ext = cubic_interp(x, y_cos, xq_oob; extrap = ExtendExtrap())
        @test vals_ext[1] ≈ ref_sin_ext
        @test vals_ext[2] ≈ ref_cos_ext

        # WrapExtrap
        vals_wrap = cubic_interp(x, Series(y_sin, y_cos), xq_oob; extrap = WrapExtrap())
        ref_wrap_sin = cubic_interp(x, y_sin, xq_oob; extrap = WrapExtrap())
        ref_wrap_cos = cubic_interp(x, y_cos, xq_oob; extrap = WrapExtrap())
        @test vals_wrap[1] ≈ ref_wrap_sin
        @test vals_wrap[2] ≈ ref_wrap_cos

        # FillExtrap
        vals_fill = cubic_interp(x, Series(y_sin, y_cos), xq_oob; extrap = FillExtrap(999.0))
        @test vals_fill[1] ≈ 999.0
        @test vals_fill[2] ≈ 999.0
    end

    @testset "Derivative ops" begin
        for d in 0:4
            op = DerivOp(d)
            vals = cubic_interp(x, Series(y_sin, y_cos), 0.37; deriv = op)
            ref_sin = cubic_interp(x, y_sin, 0.37; deriv = op)
            ref_cos = cubic_interp(x, y_cos, 0.37; deriv = op)
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
        d1 = cubic_interp(x, Series(y_sin, y_cos), 0.37; deriv = DerivOp(1))
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

    @testset "Scalar: Vector-of-Vectors" begin
        vals = cubic_interp(x, Series([y_sin, y_cos, y_exp]), 0.37)
        ref = cubic_interp(x, Series(y_sin, y_cos, y_exp), 0.37)
        @test vals ≈ ref
    end

    @testset "DimensionMismatch on wrong output size" begin
        s = Series(y_sin, y_cos)
        out_wrong = zeros(5)
        @test_throws DimensionMismatch cubic_interp!(out_wrong, x, s, 0.5)
    end
end
