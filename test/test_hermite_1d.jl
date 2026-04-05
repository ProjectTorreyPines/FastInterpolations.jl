# ALLOC_THRESHOLD is defined in runtests.jl

@testset "Cubic Hermite 1D" begin

    # ========================================
    # Test data: cubic polynomial p(x) = 2x³ - x² + 3x - 1
    # Hermite with exact derivatives should reproduce exactly.
    # ========================================
    p(x) = 2x^3 - x^2 + 3x - 1
    dp(x) = 6x^2 - 2x + 3
    d2p(x) = 12x - 2
    d3p(x::T) where {T} = T(12)

    @testset "Cubic polynomial exactness — oneshot scalar" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)

        xq_pts = [0.15, 0.7, 1.33, 2.5, 2.99]
        for xq in xq_pts
            val = hermite_interp(x, y, dy, xq)
            @test val ≈ p(xq) atol = 1.0e-12
        end
    end

    @testset "Cubic polynomial exactness — oneshot vector" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)
        xq = [0.15, 0.7, 1.33, 2.5, 2.99]

        result = hermite_interp(x, y, dy, xq)
        @test result ≈ p.(xq) atol = 1.0e-12
    end

    @testset "Cubic polynomial exactness — in-place" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)
        xq = [0.15, 0.7, 1.33, 2.5, 2.99]
        out = zeros(length(xq))

        hermite_interp!(out, x, y, dy, xq)
        @test out ≈ p.(xq) atol = 1.0e-12
    end

    @testset "Cubic polynomial exactness — callable interpolant" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)

        itp = hermite_interp(x, y, dy)
        @test itp isa CubicHermiteInterpolant1D

        xq_pts = [0.15, 0.7, 1.33, 2.5, 2.99]
        for xq in xq_pts
            @test itp(xq) ≈ p(xq) atol = 1.0e-12
        end

        # Vector call
        xq_vec = [0.15, 0.7, 1.33, 2.5, 2.99]
        @test itp(xq_vec) ≈ p.(xq_vec) atol = 1.0e-12

        # In-place call
        out = zeros(length(xq_vec))
        itp(out, xq_vec)
        @test out ≈ p.(xq_vec) atol = 1.0e-12
    end

    @testset "DerivOp correctness" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)

        xq = 1.5

        # Value
        @test hermite_interp(x, y, dy, xq; deriv = EvalValue()) ≈ p(xq) atol = 1.0e-12

        # First derivative
        @test hermite_interp(x, y, dy, xq; deriv = DerivOp(1)) ≈ dp(xq) atol = 1.0e-10

        # Second derivative
        @test hermite_interp(x, y, dy, xq; deriv = DerivOp(2)) ≈ d2p(xq) atol = 1.0e-8

        # Third derivative (constant 12 for cubic polynomial)
        @test hermite_interp(x, y, dy, xq; deriv = DerivOp(3)) ≈ d3p(xq) atol = 1.0e-6

        # Fourth+ derivative → zero
        @test hermite_interp(x, y, dy, xq; deriv = DerivOp(4)) ≈ 0.0 atol = 1.0e-14

        # Via callable
        itp = hermite_interp(x, y, dy)
        @test itp(xq; deriv = DerivOp(1)) ≈ dp(xq) atol = 1.0e-10
        @test itp(xq; deriv = DerivOp(2)) ≈ d2p(xq) atol = 1.0e-8
    end

    @testset "Extrapolation modes" begin
        # Use cubic polynomial for exact value checks
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)

        # NoExtrap — throws outside domain
        @test_throws DomainError hermite_interp(x, y, dy, -0.1)
        @test_throws DomainError hermite_interp(x, y, dy, 3.1)

        # ClampExtrap — returns boundary values (value), zero (derivatives)
        @test hermite_interp(x, y, dy, -0.5; extrap = ClampExtrap()) ≈ y[1]
        @test hermite_interp(x, y, dy, 3.5; extrap = ClampExtrap()) ≈ y[end]
        @test hermite_interp(x, y, dy, -0.5; extrap = ClampExtrap(), deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14

        # ExtendExtrap — extends boundary polynomial, exact for cubic
        @test hermite_interp(x, y, dy, 3.2; extrap = ExtendExtrap()) ≈ p(3.2) atol = 1.0e-10
        @test hermite_interp(x, y, dy, -0.3; extrap = ExtendExtrap()) ≈ p(-0.3) atol = 1.0e-10
        # ExtendExtrap derivative also exact
        @test hermite_interp(x, y, dy, 3.2; extrap = ExtendExtrap(), deriv = DerivOp(1)) ≈ dp(3.2) atol = 1.0e-8

        # FillExtrap — returns fill value outside domain
        @test hermite_interp(x, y, dy, -0.5; extrap = FillExtrap(999.0)) ≈ 999.0
        @test hermite_interp(x, y, dy, 3.5; extrap = FillExtrap(NaN)) |> isnan
        # FillExtrap derivative at OOB → zero (derivative of constant fill)
        @test hermite_interp(x, y, dy, -0.5; extrap = FillExtrap(999.0), deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14
        @test hermite_interp(x, y, dy, 3.5; extrap = FillExtrap(999.0), deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14

        # WrapExtrap — wraps to domain
        val_wrap = hermite_interp(x, y, dy, 3.2; extrap = WrapExtrap())
        @test isfinite(val_wrap)
        # WrapExtrap — vector fast-path includes x_max (no unnecessary wrap overhead)
        xq_boundary = [x[end]]
        out_boundary = similar(xq_boundary)
        hermite_interp!(out_boundary, x, y, dy, xq_boundary; extrap = WrapExtrap())
        @test isfinite(out_boundary[1])

        # Callable with extrap
        itp = hermite_interp(x, y, dy; extrap = ClampExtrap())
        @test itp(-0.5) ≈ y[1]
        @test itp(3.5) ≈ y[end]
    end

    @testset "Type stability — @inferred" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)
        xq_scalar = 0.5
        xq_vec = [0.2, 0.5, 0.8]

        # Scalar oneshot
        @test @inferred(hermite_interp(x, y, dy, xq_scalar)) isa Float64

        # Vector oneshot
        @test @inferred(hermite_interp(x, y, dy, xq_vec)) isa Vector{Float64}

        # Callable scalar
        itp = hermite_interp(x, y, dy)
        @test @inferred(itp(xq_scalar)) isa Float64

        # Callable vector
        @test @inferred(itp(xq_vec)) isa Vector{Float64}
    end

    @testset "Zero allocation" begin
        x = collect(range(0.0, 1.0, 50))
        y = sin.(x)
        dy = cos.(x)

        # Scalar oneshot — function barrier for accurate measurement
        function test_scalar_alloc(x, y, dy, xq)
            hermite_interp(x, y, dy, xq)
            return @allocated hermite_interp(x, y, dy, xq)
        end
        @test test_scalar_alloc(x, y, dy, 0.5) <= ALLOC_THRESHOLD

        # Callable scalar
        itp = hermite_interp(x, y, dy)
        function test_callable_alloc(itp, xq)
            itp(xq)
            return @allocated itp(xq)
        end
        @test test_callable_alloc(itp, 0.5) == 0
    end

    @testset "Uniform grid (Range input)" begin
        x = range(0.0, 3.0, 20)
        y = p.(collect(x))
        dy = dp.(collect(x))

        # Range grid → O(1) direct search
        @test hermite_interp(x, y, dy, 1.5) ≈ p(1.5) atol = 1.0e-12

        # Callable with range grid — verify Range preserved as _CachedRange
        itp = hermite_interp(x, y, dy)
        @test itp.x isa AbstractRange
        @test itp.x isa FastInterpolations._CachedRange
        @test itp(1.5) ≈ p(1.5) atol = 1.0e-12
    end

    @testset "Non-uniform grid" begin
        x = sort([0.0, 0.1, 0.3, 0.7, 1.2, 2.0, 2.5, 3.0])
        y = p.(x)
        dy = dp.(x)

        xq = 1.5
        @test hermite_interp(x, y, dy, xq) ≈ p(xq) atol = 1.0e-12
    end

    @testset "2-point grid (minimum)" begin
        x = [0.0, 1.0]
        y = p.(x)
        dy = dp.(x)

        xq = 0.5
        @test hermite_interp(x, y, dy, xq) ≈ p(xq) atol = 1.0e-12
    end

    @testset "Complex values" begin
        x = collect(range(0.0, 2π, 20))
        y = exp.(1im .* x)
        dy = 1im .* exp.(1im .* x)

        itp = hermite_interp(x, y, dy)
        xq = 1.0
        @test itp(xq) ≈ exp(1im * xq) atol = 1.0e-4
    end

    @testset "Integer input promotion" begin
        x_int = 0:10
        y_int = collect(x_int) .^ 2
        dy_int = 2 .* collect(x_int)

        # Should auto-promote to Float
        val = hermite_interp(collect(x_int), y_int, dy_int, 5.5)
        @test val ≈ 5.5^2 atol = 1.0e-10
    end

    @testset "Error conditions" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)

        # Length mismatch between y and dy
        @test_throws ArgumentError CubicHermiteInterpolant1D(x, y, cos.(x[1:5]); extrap = NoExtrap())
    end

    @testset "Hermite matches spline on cubic polynomial data" begin
        # For a true cubic polynomial, both spline and Hermite should agree
        x = collect(range(0.0, 3.0, 15))
        y = p.(x)
        dy = dp.(x)
        xq = [0.3, 1.0, 1.7, 2.5]

        hermite_vals = hermite_interp(x, y, dy, xq)
        spline_vals = cubic_interp(x, y, xq)

        # Both should reproduce the cubic polynomial exactly
        @test hermite_vals ≈ p.(xq) atol = 1.0e-10
        @test spline_vals ≈ p.(xq) atol = 1.0e-10
    end

    @testset "Float32 precision" begin
        x32 = collect(range(0.0f0, 3.0f0, 20))
        y32 = Float32.(p.(x32))
        dy32 = Float32.(dp.(x32))

        val = hermite_interp(x32, y32, dy32, 1.5f0)
        @test val isa Float32
        @test val ≈ Float32(p(1.5)) atol = 1.0e-4  # Float32 precision

        itp32 = hermite_interp(x32, y32, dy32)
        @test itp32(1.5f0) isa Float32
        @test itp32(1.5f0) ≈ Float32(p(1.5)) atol = 1.0e-4
    end

    @testset "Mixed precision — y::Float32, dy::Float64" begin
        x = collect(range(0.0, 3.0, 20))
        y32 = Float32.(p.(x))
        dy64 = dp.(x)  # Float64

        # Joint promotion: dy::Float64 widens Tg to Float64
        val = hermite_interp(x, y32, dy64, 1.5)
        @test val isa Float64
        @test val ≈ p(1.5) atol = 1.0e-6

        itp = hermite_interp(x, y32, dy64)
        @test itp(1.5) isa Float64
    end

    @testset "Mixed precision — x::Float32, dy::Float64" begin
        x32 = collect(range(0.0f0, 3.0f0, 20))
        y32 = Float32.(p.(x32))
        dy64 = Float64.(dp.(x32))

        # dy::Float64 widens Tg from Float32 → Float64
        val = hermite_interp(x32, y32, dy64, 1.5)
        @test val isa Float64
    end

    @testset "Complex values — hermite_interp" begin
        x = collect(range(0.0, 2π, 30))
        y_c = complex.(sin.(x), cos.(x))
        dy_c = complex.(cos.(x), -sin.(x))

        val = hermite_interp(x, y_c, dy_c, 1.0)
        @test val isa ComplexF64
        @test real(val) ≈ sin(1.0) atol = 0.01
        @test imag(val) ≈ cos(1.0) atol = 0.01

        itp = hermite_interp(x, y_c, dy_c)
        @test itp(1.0) isa ComplexF64
        @test itp(1.0) ≈ val
    end

    @testset "Broadcast" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)

        itp = hermite_interp(x, y, dy)
        xq = [0.5, 1.0, 2.0]
        broadcast_result = itp.(xq)
        @test broadcast_result ≈ p.(xq) atol = 1.0e-12
    end

    @testset "Search policy override" begin
        x = collect(range(0.0, 3.0, 50))
        y = p.(x)
        dy = dp.(x)
        xq = 1.5

        # Oneshot with explicit search
        @test hermite_interp(x, y, dy, xq; search = BinarySearch()) ≈ p(xq) atol = 1.0e-12
        @test hermite_interp(x, y, dy, xq; search = LinearBinarySearch()) ≈ p(xq) atol = 1.0e-12

        # Callable with search override at call time
        itp = hermite_interp(x, y, dy)
        @test itp(xq; search = BinarySearch()) ≈ p(xq) atol = 1.0e-12
    end

    @testset "Show methods" begin
        x = collect(range(0.0, 3.0, 10))
        y = sin.(x)
        dy = cos.(x)
        itp = hermite_interp(x, y, dy)

        # Compact
        compact = sprint(show, itp)
        @test occursin("CubicHermiteInterpolant1D", compact)
        @test occursin("user slopes", compact)

        # Verbose
        verbose = sprint(show, MIME"text/plain"(), itp)
        @test occursin("CubicHermiteInterpolant1D", verbose)
        @test occursin("user-supplied", verbose)
    end

    @testset "Duck typing — minimal DuckFloat5" begin
        # 5-op type: +(Tv,Tv), -(Tv,Tv), *(Real,Tv), *(Tv,Real)
        # Same minimal contract as the comprehensive duck typing suite
        struct HermiteDuck
            v::Float64
        end
        Base.:+(a::HermiteDuck, b::HermiteDuck) = HermiteDuck(a.v + b.v)
        Base.:-(a::HermiteDuck, b::HermiteDuck) = HermiteDuck(a.v - b.v)
        Base.:*(a::Real, b::HermiteDuck) = HermiteDuck(a * b.v)
        Base.:*(a::HermiteDuck, b::Real) = HermiteDuck(a.v * b)
        _dval(d::HermiteDuck) = d.v

        x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        y_raw = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        dy_raw = [0.5, 1.0, -0.5, 0.8, -0.3, 0.7]

        y_duck = HermiteDuck.(y_raw)
        dy_duck = HermiteDuck.(dy_raw)

        # Reference: plain Float64
        ref = hermite_interp(x, y_raw, dy_raw, 2.5)

        # Oneshot scalar
        duck_val = hermite_interp(x, y_duck, dy_duck, 2.5)
        @test duck_val isa HermiteDuck
        @test _dval(duck_val) ≈ ref

        # Oneshot vector
        xq_vec = [1.5, 2.5, 3.5]
        ref_vec = hermite_interp(x, y_raw, dy_raw, xq_vec)
        duck_vec = hermite_interp(x, y_duck, dy_duck, xq_vec)
        @test all(d -> d isa HermiteDuck, duck_vec)
        @test _dval.(duck_vec) ≈ ref_vec

        # Callable interpolant
        itp = hermite_interp(x, y_duck, dy_duck)
        @test itp(2.5) isa HermiteDuck
        @test _dval(itp(2.5)) ≈ ref

        # DerivOp(1)
        ref_d1 = hermite_interp(x, y_raw, dy_raw, 2.5; deriv = DerivOp(1))
        duck_d1 = hermite_interp(x, y_duck, dy_duck, 2.5; deriv = DerivOp(1))
        @test duck_d1 isa HermiteDuck
        @test _dval(duck_d1) ≈ ref_d1

        # Type stability
        @test @inferred(hermite_interp(x, y_duck, dy_duck, 2.5)) isa HermiteDuck
    end

    @testset "Duck typing — SVector values" begin
        using StaticArrays

        x = collect(range(0.0, 5.0, 10))
        # SVector{3}-valued function and derivatives
        y_sv = [SVector(sin(xi), cos(xi), xi^2) for xi in x]
        dy_sv = [SVector(cos(xi), -sin(xi), 2xi) for xi in x]

        xq = 2.3

        # Oneshot scalar
        val = hermite_interp(x, y_sv, dy_sv, xq)
        @test val isa SVector{3, Float64}
        @test val[1] ≈ sin(xq) atol = 1.0e-3
        @test val[3] ≈ xq^2 atol = 1.0e-3

        # Callable interpolant
        itp = hermite_interp(x, y_sv, dy_sv)
        @test itp(xq) isa SVector{3, Float64}
        @test itp(xq)[1] ≈ sin(xq) atol = 1.0e-3

        # DerivOp(1) — should return SVector of derivatives
        d1 = hermite_interp(x, y_sv, dy_sv, xq; deriv = DerivOp(1))
        @test d1 isa SVector{3, Float64}
        @test d1[1] ≈ cos(xq) atol = 1.0e-2

        # Type stability
        @test @inferred(hermite_interp(x, y_sv, dy_sv, xq)) isa SVector{3, Float64}
    end

    @testset "In-place hint forwarding and zero-alloc" begin
        x = collect(range(0.0, 4.0, 50))
        y = sin.(x)
        dy = cos.(x)
        xq = collect(range(0.1, 3.9, 20))
        output = Vector{Float64}(undef, length(xq))

        # hint gets updated through in-place path
        hint = Ref(1)
        hermite_interp!(output, x, y, dy, xq; hint)
        @test hint[] > 1  # hint updated after sorted queries

        # Verify correctness with hint
        ref = hermite_interp(x, y, dy, xq)
        @test output ≈ ref

        # Zero-allocation in-place (function barrier)
        function test_inplace_alloc(output, x, y, dy, xq)
            hint_local = Ref(1)
            hermite_interp!(output, x, y, dy, xq; hint = hint_local)
            return @allocated hermite_interp!(output, x, y, dy, xq; hint = hint_local)
        end
        @test test_inplace_alloc(output, x, y, dy, xq) <= ALLOC_THRESHOLD
    end

    @testset "Coverage — WrapExtrap vector paths" begin
        x = collect(range(0.0, 2π, 20))
        y = sin.(x)
        dy = cos.(x)
        # Vector fast-path: all queries strictly within (x_min, x_max)
        xq_inner = collect(range(0.1, 2π - 0.1, 10))
        out = similar(xq_inner)
        hermite_interp!(out, x, y, dy, xq_inner; extrap = WrapExtrap())
        @test all(isfinite, out)

        # Vector slow-path: some queries outside → per-element wrap
        xq_cross = [0.5, 2π + 0.3, -0.2]
        result = hermite_interp(x, y, dy, xq_cross; extrap = WrapExtrap())
        @test all(isfinite, result)

        # Interpolant vector path with WrapExtrap
        itp = hermite_interp(x, y, dy; extrap = WrapExtrap())
        out2 = itp(xq_inner)
        @test all(isfinite, out2)
    end

    @testset "Coverage — Range in-place disambiguation" begin
        x_range = range(0.0, 3.0, 20)
        y = sin.(collect(x_range))
        dy = cos.(collect(x_range))
        xq = collect(range(0.1, 2.9, 10))
        out = similar(xq)

        # Calls the AbstractRange{Tg} overload of hermite_interp!
        hermite_interp!(out, x_range, y, dy, xq)
        @test out ≈ sin.(xq) atol = 0.1
    end

    @testset "Coverage — vector DerivOp(2+)" begin
        x = collect(range(0.0, 3.0, 30))
        p(t) = 2t^3 + t^2 - t + 1
        dp(t) = 6t^2 + 2t - 1
        d2p(t) = 12t + 2
        y = p.(x)
        dy = dp.(x)
        xq = [0.5, 1.0, 2.0]

        # DerivOp(2) through vector path
        result2 = hermite_interp(x, y, dy, xq; deriv = DerivOp(2))
        @test result2 ≈ d2p.(xq) atol = 1.0e-6

        # DerivOp(3) through vector path
        result3 = hermite_interp(x, y, dy, xq; deriv = DerivOp(3))
        @test all(r -> abs(r - 12.0) < 1.0e-3, result3)  # constant 12 for cubic
    end

    @testset "Coverage — show with Range grid" begin
        x_range = range(0.0, 3.0, 10)
        y = sin.(collect(x_range))
        dy = cos.(collect(x_range))
        itp = hermite_interp(x_range, y, dy)

        verbose = sprint(show, MIME"text/plain"(), itp)
        @test occursin("CubicHermiteInterpolant1D", verbose)
        # Range grid → no Search row
        @test !occursin("Search:", verbose)
    end

    @testset "Coverage — ClampExtrap in-domain scalar (Vector + Range grid)" begin
        # Exercises _hermite_eval_at_point ClampOrFill in-domain path (no-spacing)
        x = collect(range(0.0, 3.0, 20))
        y = sin.(x)
        dy = cos.(x)
        # Query inside domain with ClampExtrap → takes the kernel path, not the boundary return
        @test hermite_interp(x, y, dy, 1.5; extrap = ClampExtrap()) ≈ sin(1.5) atol = 0.01

        # Same with Range grid (spacing overload) via callable
        x_range = range(0.0, 3.0, 20)
        itp = hermite_interp(x_range, y, dy; extrap = ClampExtrap())
        @test itp(1.5) ≈ sin(1.5) atol = 0.01
    end

    @testset "Coverage — callable Range + WrapExtrap OOB vector" begin
        # Exercises spacing-based WrapExtrap vector slow-path (lines 228-230)
        x_range = range(0.0, 2π, 20)
        y = sin.(collect(x_range))
        dy = cos.(collect(x_range))
        itp = hermite_interp(x_range, y, dy; extrap = WrapExtrap())
        # Query with some OOB points → triggers slow path (per-element wrap)
        xq_oob = [0.5, 2π + 0.3, -0.2]
        result = itp(xq_oob)
        @test all(isfinite, result)
    end

    @testset "Coverage — Hermite output eltype validation error" begin
        x_int = collect(0:5)
        y_int = x_int .^ 2
        dy_int = 2 .* x_int
        out_narrow = Vector{Float32}(undef, 3)
        @test_throws ArgumentError hermite_interp!(out_narrow, x_int, y_int, dy_int, [1, 2, 3])
    end

    @testset "Coverage — generic wrapper Integer promotion" begin
        x_int = collect(0:5)
        y_int = x_int .^ 2
        dy_int = 2 .* x_int

        # Vector allocating through generic wrapper (Int → Float64)
        result = hermite_interp(x_int, y_int, dy_int, [1.5, 2.5])
        @test result ≈ [1.5^2, 2.5^2] atol = 1.0e-10

        # In-place through generic wrapper
        out = Vector{Float64}(undef, 2)
        hermite_interp!(out, x_int, y_int, dy_int, [1.5, 2.5])
        @test out ≈ [1.5^2, 2.5^2] atol = 1.0e-10
    end
end
