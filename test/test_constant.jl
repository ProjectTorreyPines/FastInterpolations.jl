# Tests for constant (step/piecewise constant) interpolation

# ============================================================================
# Group 1: Infrastructure Tests (Phase 1)
# ============================================================================
@testitem "Constant Interpolation - Infrastructure" begin
    import FastInterpolations: _constant_kernel, EvalValue, EvalDeriv1, EvalDeriv2

    @testset "AbstractSide type hierarchy" begin
        @testset "AbstractSide is abstract" begin
            @test isabstracttype(FastInterpolations.AbstractSide)
        end

        @testset "concrete subtypes" begin
            @test NearestSide() isa FastInterpolations.AbstractSide
            @test LeftSide() isa FastInterpolations.AbstractSide
            @test RightSide() isa FastInterpolations.AbstractSide
        end

        @testset "singleton identity" begin
            @test NearestSide() === NearestSide()
            @test LeftSide() === LeftSide()
            @test RightSide() === RightSide()
        end

        @testset "type distinctness" begin
            @test typeof(NearestSide()) !== typeof(LeftSide())
            @test typeof(LeftSide()) !== typeof(RightSide())
            @test typeof(NearestSide()) !== typeof(RightSide())
        end
    end

end

# ============================================================================
# Group 2: Kernel Tests (Phase 2)
# ============================================================================
@testitem "Constant Interpolation - Kernels" begin
    import FastInterpolations: _constant_kernel, EvalValue, EvalDeriv1, EvalDeriv2

    # Test setup: interval [0, 1] with y_left=10.0, y_right=20.0
    y_left = 10.0
    y_right = 20.0
    h = 1.0

    @testset "EvalValue - side=LeftSide()" begin
        op = EvalValue()
        sv = LeftSide()
        @test _constant_kernel(op, y_left, y_right, h, 0.0, sv) == 10.0
        @test _constant_kernel(op, y_left, y_right, h, 0.3, sv) == 10.0
        @test _constant_kernel(op, y_left, y_right, h, 0.7, sv) == 10.0
        @test _constant_kernel(op, y_left, y_right, h, 0.99, sv) == 10.0
    end

    @testset "EvalValue - side=RightSide()" begin
        op = EvalValue()
        sv = RightSide()
        @test _constant_kernel(op, y_left, y_right, h, 0.0, sv) == 10.0  # grid point
        @test _constant_kernel(op, y_left, y_right, h, 0.3, sv) == 20.0
        @test _constant_kernel(op, y_left, y_right, h, 0.7, sv) == 20.0
        @test _constant_kernel(op, y_left, y_right, h, 0.99, sv) == 20.0
    end

    @testset "EvalValue - side=NearestSide()" begin
        op = EvalValue()
        sv = NearestSide()
        @test _constant_kernel(op, y_left, y_right, h, 0.0, sv) == 10.0
        @test _constant_kernel(op, y_left, y_right, h, 0.4, sv) == 10.0
        @test _constant_kernel(op, y_left, y_right, h, 0.5, sv) == 10.0  # tie-breaking
        @test _constant_kernel(op, y_left, y_right, h, 0.6, sv) == 20.0
        @test _constant_kernel(op, y_left, y_right, h, 0.99, sv) == 20.0
    end

    @testset "EvalDeriv1 - all sides return zero" begin
        op = EvalDeriv1()
        @test _constant_kernel(op, y_left, y_right, h, 0.5, NearestSide()) == 0.0
        @test _constant_kernel(op, y_left, y_right, h, 0.5, LeftSide()) == 0.0
        @test _constant_kernel(op, y_left, y_right, h, 0.5, RightSide()) == 0.0
    end

    @testset "EvalDeriv2 - all sides return zero" begin
        op = EvalDeriv2()
        @test _constant_kernel(op, y_left, y_right, h, 0.5, NearestSide()) == 0.0
        @test _constant_kernel(op, y_left, y_right, h, 0.5, LeftSide()) == 0.0
        @test _constant_kernel(op, y_left, y_right, h, 0.5, RightSide()) == 0.0
    end

    @testset "Type preservation" begin
        @test _constant_kernel(EvalValue(), 1.0, 2.0, 1.0, 0.5, LeftSide()) isa Float64
        @test _constant_kernel(EvalDeriv1(), 1.0, 2.0, 1.0, 0.5, LeftSide()) isa Float64
        @test _constant_kernel(EvalValue(), 1.0f0, 2.0f0, 1.0f0, 0.5f0, NearestSide()) isa Float32
        @test _constant_kernel(EvalDeriv1(), 1.0f0, 2.0f0, 1.0f0, 0.5f0, NearestSide()) isa Float32
    end

    @testset "Non-uniform grid (h != 1)" begin
        op = EvalValue()
        @test _constant_kernel(op, 100.0, 200.0, 2.0, 0.9, NearestSide()) == 100.0
        @test _constant_kernel(op, 100.0, 200.0, 2.0, 1.0, NearestSide()) == 100.0  # tie
        @test _constant_kernel(op, 100.0, 200.0, 2.0, 1.1, NearestSide()) == 200.0
    end

end

# ============================================================================
# Group 3: API Tests (Phase 3)
# ============================================================================
@testitem "Constant Interpolation - API" begin
    import FastInterpolations: _constant_kernel, EvalValue, EvalDeriv1, EvalDeriv2

    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]

    @testset "Scalar API" begin
        @test constant_interp(x, y, 0.5) == 10.0
        @test constant_interp(x, y, 1.5) == 20.0
        @test constant_interp(x, y, 2.5) == 30.0

        @test constant_interp(x, y, 0.9; side = LeftSide()) == 10.0
        @test constant_interp(x, y, 1.1; side = LeftSide()) == 20.0

        @test constant_interp(x, y, 0.0; side = RightSide()) == 10.0
        @test constant_interp(x, y, 0.1; side = RightSide()) == 20.0

        @test constant_interp(x, y, 1.0; side = NearestSide()) == 20.0
        @test constant_interp(x, y, 1.0; side = LeftSide()) == 20.0
        @test constant_interp(x, y, 1.0; side = RightSide()) == 20.0

        @test constant_interp(x, y, 4.0) == 50.0

        @test constant_interp(x, y, 1.5; deriv = DerivOp(1)) == 0.0
        @test constant_interp(x, y, 2.5; deriv = DerivOp(2)) == 0.0
    end

    @testset "Vector API (allocating)" begin
        result = constant_interp(x, y, [0.5, 1.5, 2.5])
        @test result ≈ [10.0, 20.0, 30.0]

        result_left = constant_interp(x, y, [0.5, 1.5]; side = LeftSide())
        @test result_left ≈ [10.0, 20.0]
    end

    @testset "In-place API" begin
        out = zeros(3)
        constant_interp!(out, x, y, [0.5, 1.5, 2.5])
        @test out ≈ [10.0, 20.0, 30.0]
    end

    @testset "Extrapolation modes" begin
        @test_throws DomainError constant_interp(x, y, -1.0; extrap = NoExtrap())
        @test_throws DomainError constant_interp(x, y, 5.0; extrap = NoExtrap())

        @test constant_interp(x, y, -1.0; extrap = ClampExtrap()) == 10.0
        @test constant_interp(x, y, 5.0; extrap = ClampExtrap()) == 50.0

        @test constant_interp(x, y, -1.0; extrap = ExtendExtrap()) == 10.0
        @test constant_interp(x, y, 5.0; extrap = ExtendExtrap()) == 50.0

        # Closed `[first(x), last(x)]` (PR refac/wrap_closed): exact last(x) is in-domain,
        # returns y[end]. Strictly-OOB queries still wrap.
        @test constant_interp(x, y, 4.0; extrap = WrapExtrap()) == 50.0
        @test constant_interp(x, y, 4.5; extrap = WrapExtrap()) == 10.0

        @test constant_interp(x, y, -1.0; extrap = ClampExtrap(), deriv = DerivOp(1)) == 0.0
    end

    @testset "Real type wrapper (Integer input)" begin
        x_int = [0, 1, 2, 3, 4]
        y_int = [10, 20, 30, 40, 50]
        # Constant duck-types output to `eltype(y)` — Int in → Int out (no Float
        # widening, since the kernel is selection, not arithmetic).
        result = constant_interp(x_int, y_int, 1.5)
        @test result isa Int
        @test result == 20
    end

    @testset "Real→Float wrappers (coverage)" begin
        # Integer grid data (T<:Real, not T<:AbstractFloat)
        x_int = [0, 1, 2, 3, 4]
        y_int = [10, 20, 30, 40, 50]

        # Test 1: ConstantInterpolant scalar with Integer input (lines 167-169)
        # This tests itp(xi::S) where S<:Real, S≠T
        itp = constant_interp([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0, 50.0])
        @test itp(1) == 20.0  # Integer query to Float64 interpolant
        @test itp(2) == 30.0
        @test itp(0) == 10.0

        # Test 2: ConstantInterpolant in-place with type conversion (lines 201-211)
        # This tests itp(output, xi::AbstractVector{S}) where S≠T
        out = zeros(3)
        itp(out, [0, 1, 2])  # Integer query vector
        @test out ≈ [10.0, 20.0, 30.0]

        # Test 3: Vector allocating with Integer grid + Integer query.
        # Constant duck-types output to `eltype(y)` — no Float widening.
        result_vec = constant_interp(x_int, y_int, [0, 1, 2])
        @test result_vec isa Vector{Int}
        @test result_vec == [10, 20, 30]

        # Test 4: In-place Real→Float wrapper (lines 440-468)
        # constant_interp!(output, x::AbstractVector{T}, y::AbstractVector{T}, x_targets::AbstractVector{S})
        out2 = zeros(3)
        constant_interp!(out2, x_int, y_int, [0, 1, 2])
        @test out2 ≈ [10.0, 20.0, 30.0]

        # Test 5: 2-arg callable Real→Float wrapper (lines 474-482)
        # constant_interp(x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
        itp_int = constant_interp(x_int, y_int)
        @test itp_int isa FastInterpolations.ConstantInterpolant
        @test itp_int(0.5) == 10.0
        @test itp_int(1.5) == 20.0

        # Test options pass through the wrappers correctly
        @test constant_interp(x_int, y_int, 0; side = RightSide()) == 10.0
        @test constant_interp(x_int, y_int, 0; side = LeftSide()) == 10.0
        @test constant_interp(x_int, y_int, -1; extrap = ClampExtrap()) == 10.0
        @test constant_interp(x_int, y_int, 5; extrap = ClampExtrap()) == 50.0

        # Test in-place wrapper with options
        out3 = zeros(2)
        constant_interp!(out3, x_int, y_int, [-1, 5]; extrap = ClampExtrap())
        @test out3 ≈ [10.0, 50.0]

        # Test 2-arg callable with options
        itp_left = constant_interp(x_int, y_int; side = LeftSide())
        @test itp_left(0.9) == 10.0
        itp_wrap = constant_interp(x_int, y_int; extrap = WrapExtrap())
        # Closed-domain: xq == last(x) returns y[end]; was y[1] under prior half-open.
        @test itp_wrap(4.0) == 50
    end

    @testset "ConstantInterpolant - 2-arg form" begin
        itp = constant_interp(x, y)
        @test itp isa FastInterpolations.ConstantInterpolant
    end

    @testset "ConstantInterpolant - Scalar call" begin
        itp = constant_interp(x, y)
        @test itp(0.5) == 10.0
        @test itp(1.5) == 20.0
        @test itp(4.0) == 50.0
        @test itp(1.5; deriv = DerivOp(1)) == 0.0
        @test itp(2.5; deriv = DerivOp(2)) == 0.0
    end

    @testset "ConstantInterpolant - Vector call" begin
        itp = constant_interp(x, y)
        result = itp([0.5, 1.5, 2.5])
        @test result ≈ [10.0, 20.0, 30.0]
    end

    @testset "ConstantInterpolant - Broadcast fusion" begin
        itp = constant_interp(x, y)
        result = itp.([0.5, 1.5])
        @test result ≈ [10.0, 20.0]

        coef = 2.0
        query = [0.5, 1.5]
        result_fused = @. coef * itp(query)
        @test result_fused ≈ [20.0, 40.0]
    end

    @testset "ConstantInterpolant - Options" begin
        itp_wrap = constant_interp(x, y; extrap = WrapExtrap())
        # Closed-domain: xq == last(x) returns y[end].
        @test itp_wrap(4.0) == 50.0

        itp_const = constant_interp(x, y; extrap = ClampExtrap())
        @test itp_const(-1.0) == 10.0
        @test itp_const(5.0) == 50.0

        itp_left = constant_interp(x, y; side = LeftSide())
        @test itp_left(0.9) == 10.0

        itp_right = constant_interp(x, y; side = RightSide())
        @test itp_right(0.1) == 20.0
    end

    @testset "Type inference" begin
        itp = constant_interp(x, y)
        @test @inferred(itp(0.5)) isa Float64
        @test @inferred(constant_interp(x, y, 0.5)) isa Float64
        @test @inferred(constant_interp(x, y, 0.5; side = LeftSide())) isa Float64
        @test @inferred(constant_interp(x, y, 0.5; extrap = ClampExtrap())) isa Float64
        @test @inferred(constant_interp(x, y, 0.5; deriv = DerivOp(1))) isa Float64
    end

    @testset "Invalid side argument throws TypeError" begin
        @test_throws TypeError constant_interp(x, y, 0.5; side = :invalid)
        @test_throws TypeError constant_interp(x, y, [0.5, 1.5]; side = :foo)
        out = zeros(2)
        @test_throws TypeError constant_interp!(out, x, y, [0.5, 1.5]; side = :bar)
    end

    @testset "Direct outer kwarg ctor" begin
        # Cover `ConstantInterpolant(x, y; ...)` direct call — the factory
        # `constant_interp` path is exercised above; this exercises the
        # outer kwarg ctor's `_cache_axis` + `bc` forwarding lines.
        x_vec = [0.0, 1.0, 2.0, 3.0]
        y_vec = [10.0, 20.0, 30.0, 40.0]
        x_rng = range(0.0, 3.0; length = 4)

        itp_v = ConstantInterpolant(x_vec, y_vec)
        @test itp_v.x isa FastInterpolations._CachedVector
        @test itp_v(0.5) == 10.0

        itp_r = ConstantInterpolant(x_rng, y_vec)
        @test itp_r.x isa FastInterpolations._CachedRange

        itp_full = ConstantInterpolant(
            x_vec, y_vec;
            extrap = ClampExtrap(), side = LeftSide(), search = BinarySearch()
        )
        @test itp_full(-1.0) == 10.0  # clamped left
        @test itp_full(0.5) == 10.0   # LeftSide
    end

end

# ============================================================================
# Group 4: Allocation Tests (Phase 3)
# ============================================================================
@testitem "Constant Interpolation - Allocations" setup = [AllocConstants] begin
    import FastInterpolations: _constant_kernel, EvalValue, EvalDeriv1, EvalDeriv2

    x = collect(0.0:0.1:1.0)
    y = sin.(x)

    @testset "constant_interp! in-place" begin
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        constant_interp!(out, x, y, xq)
        allocs = @allocated constant_interp!(out, x, y, xq)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "constant_interp! with side option" begin
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        constant_interp!(out, x, y, xq; side = LeftSide())
        allocs = @allocated constant_interp!(out, x, y, xq; side = LeftSide())
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "constant_interp! with extrap option" begin
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        constant_interp!(out, x, y, xq; extrap = ClampExtrap())
        allocs = @allocated constant_interp!(out, x, y, xq; extrap = ClampExtrap())
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "ConstantInterpolant scalar" begin
        itp = constant_interp(x, y)
        itp(0.5)
        allocs = @allocated itp(0.55)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "ConstantInterpolant in-place vector" begin
        itp = constant_interp(x, y)
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        itp(out, xq)
        allocs = @allocated itp(out, xq)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "ConstantInterpolant in-place with deriv" begin
        itp = constant_interp(x, y)
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        itp(out, xq; deriv = DerivOp(1))
        allocs = @allocated itp(out, xq; deriv = DerivOp(1))
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "ConstantInterpolant with extrap=WrapExtrap() in-place" begin
        itp = constant_interp(x, y; extrap = WrapExtrap())
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        itp(out, xq)
        allocs = @allocated itp(out, xq)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "ConstantInterpolant with side=LeftSide() in-place" begin
        itp = constant_interp(x, y; side = LeftSide())
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        itp(out, xq)
        allocs = @allocated itp(out, xq)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ─────────────────────────────────────────────────────────────
    # Real→Float wrapper zero-allocation tests
    # When types already match Float64, _to_float identity should be used
    # ─────────────────────────────────────────────────────────────

    @testset "Real wrapper - constant_interp! when types match" begin
        # When x, y are Float64 and x_targets is Float64, Real wrapper
        # should be zero-alloc due to _to_float identity specialization
        x_f64 = collect(0.0:0.1:1.0)
        y_f64 = sin.(x_f64)
        xq_f64 = [0.1, 0.3, 0.5, 0.7, 0.9]
        out = zeros(5)

        # Warmup - ensure compilation
        constant_interp!(out, x_f64, y_f64, xq_f64)

        allocs = @allocated constant_interp!(out, x_f64, y_f64, xq_f64)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Real wrapper - ConstantInterpolant in-place when types match" begin
        x_f64 = collect(0.0:0.1:1.0)
        y_f64 = sin.(x_f64)
        itp = constant_interp(x_f64, y_f64)
        xq_f64 = [0.1, 0.3, 0.5, 0.7, 0.9]
        out = zeros(5)

        # Warmup
        itp(out, xq_f64)

        allocs = @allocated itp(out, xq_f64)
        @test allocs <= ALLOC_THRESHOLD
    end

end

# ============================================================================
# Group 5: Integration & Edge Cases (Phase 4)
# ============================================================================
@testitem "Constant Interpolation - Integration" begin
    import FastInterpolations: _constant_kernel, EvalValue, EvalDeriv1, EvalDeriv2

    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]

    @testset "DerivativeView - deriv1" begin
        itp = constant_interp(x, y)
        d1 = deriv1(itp)
        @test d1 isa FastInterpolations.DerivativeView{1}
        @test d1(0.5) == 0.0
        @test d1(1.5) == 0.0
        @test d1(2.5) == 0.0
        @test d1(3.5) == 0.0

        result = d1.([0.5, 1.5, 2.5])
        @test result ≈ [0.0, 0.0, 0.0]
    end

    @testset "DerivativeView - deriv2" begin
        itp = constant_interp(x, y)
        d2 = deriv2(itp)
        @test d2 isa FastInterpolations.DerivativeView{2}
        @test d2(0.5) == 0.0
        @test d2(1.5) == 0.0

        result = d2.([0.5, 1.5, 2.5])
        @test result ≈ [0.0, 0.0, 0.0]
    end

    @testset "DerivativeView - fused broadcast" begin
        itp = constant_interp(x, y)
        d1 = deriv1(itp)
        coef = 2.0
        xs = [0.5, 1.5, 2.5]
        result = @. coef * d1(xs)
        @test result ≈ [0.0, 0.0, 0.0]
    end

    @testset "DerivativeView - type stability" begin
        itp = constant_interp(x, y)
        d1 = deriv1(itp)
        d2 = deriv2(itp)
        @test @inferred(d1(0.5)) isa Float64
        @test @inferred(d2(0.5)) isa Float64
    end

    @testset "DerivativeView - with options" begin
        itp_const = constant_interp(x, y; extrap = ClampExtrap())
        d1 = deriv1(itp_const)
        @test d1(-1.0) == 0.0
        @test d1(5.0) == 0.0

        itp_wrap = constant_interp(x, y; extrap = WrapExtrap())
        d2 = deriv2(itp_wrap)
        @test d2(4.5) == 0.0

        itp_left = constant_interp(x, y; side = LeftSide())
        @test deriv1(itp_left)(0.5) == 0.0

        itp_right = constant_interp(x, y; side = RightSide())
        @test deriv2(itp_right)(0.5) == 0.0
    end

    @testset "Edge: Grid point behavior" begin
        for side in (NearestSide(), LeftSide(), RightSide())
            @test constant_interp(x, y, 0.0; side = side) == 10.0
            @test constant_interp(x, y, 1.0; side = side) == 20.0
            @test constant_interp(x, y, 2.0; side = side) == 30.0
            @test constant_interp(x, y, 3.0; side = side) == 40.0
            @test constant_interp(x, y, 4.0; side = side) == 50.0
        end
    end

    @testset "Edge: xi == x[end] with extrap modes" begin
        # Closed-domain (PR refac/wrap_closed): every extrap policy now agrees
        # at the exact right boundary — xq == last(x) returns y[end].
        @test constant_interp(x, y, 4.0; extrap = NoExtrap()) == 50.0
        @test constant_interp(x, y, 4.0; extrap = ClampExtrap()) == 50.0
        @test constant_interp(x, y, 4.0; extrap = ExtendExtrap()) == 50.0
        @test constant_interp(x, y, 4.0; extrap = WrapExtrap()) == 50.0
    end

    @testset "Edge: Wrap boundary cases" begin
        # Closed `[first(x), last(x)]`: xq == last(x) returns y[end] via the
        # x_last short-circuit in `_constant_eval_at_anchor` (which now uses
        # `y[aq.idxR]`; idxR == n for non-periodic). Side flag is irrelevant
        # at the exact right edge because both LeftSide/RightSide/NearestSide
        # collapse onto the single boundary point.
        @test constant_interp(x, y, 4.0; extrap = WrapExtrap(), side = LeftSide()) == 50.0
        @test constant_interp(x, y, 4.0; extrap = WrapExtrap(), side = RightSide()) == 50.0
        @test constant_interp(x, y, 4.0; extrap = WrapExtrap(), side = NearestSide()) == 50.0
        # Strictly-OOB queries still wrap as before.
        @test constant_interp(x, y, 4.5; extrap = WrapExtrap(), side = NearestSide()) == 10.0
        @test constant_interp(x, y, 5.0; extrap = WrapExtrap(), side = NearestSide()) == 20.0
    end

    @testset "Edge: Extrapolation + deriv" begin
        @test constant_interp(x, y, -1.0; extrap = ClampExtrap(), deriv = DerivOp(1)) == 0.0
        @test constant_interp(x, y, 5.0; extrap = ClampExtrap(), deriv = DerivOp(2)) == 0.0
        @test constant_interp(x, y, -1.0; extrap = ExtendExtrap(), deriv = DerivOp(1)) == 0.0
        @test constant_interp(x, y, 5.0; extrap = ExtendExtrap(), deriv = DerivOp(2)) == 0.0
        @test constant_interp(x, y, 4.5; extrap = WrapExtrap(), deriv = DerivOp(1)) == 0.0
    end

    @testset "Edge: Midpoint tie-breaking" begin
        @test constant_interp(x, y, 0.5; side = NearestSide()) == 10.0
        @test constant_interp(x, y, 1.5; side = NearestSide()) == 20.0
        @test constant_interp(x, y, 2.5; side = NearestSide()) == 30.0
    end

    @testset "Edge: Non-uniform grid" begin
        x_nu = [0.0, 0.5, 2.0, 2.5, 4.0]
        y_nu = [10.0, 20.0, 30.0, 40.0, 50.0]
        @test constant_interp(x_nu, y_nu, 0.25; side = NearestSide()) == 10.0
        @test constant_interp(x_nu, y_nu, 1.0; side = NearestSide()) == 20.0
        @test constant_interp(x_nu, y_nu, 1.5; side = NearestSide()) == 30.0
    end

    @testset "Edge: Float32 type preservation" begin
        x32 = Float32[0.0, 1.0, 2.0, 3.0, 4.0]
        y32 = Float32[10.0, 20.0, 30.0, 40.0, 50.0]

        result = constant_interp(x32, y32, 0.5f0)
        @test result isa Float32
        @test result == 10.0f0

        itp32 = constant_interp(x32, y32)
        @test itp32(0.5f0) isa Float32
        @test itp32(0.5f0) == 10.0f0
    end

    @testset "Edge: Minimum grid size (2 points)" begin
        x_min = [0.0, 1.0]
        y_min = [10.0, 20.0]
        @test constant_interp(x_min, y_min, 0.5; side = NearestSide()) == 10.0
        @test constant_interp(x_min, y_min, 0.5; side = LeftSide()) == 10.0
        @test constant_interp(x_min, y_min, 0.5; side = RightSide()) == 20.0
        @test constant_interp(x_min, y_min, 0.0) == 10.0
        @test constant_interp(x_min, y_min, 1.0) == 20.0
    end

    @testset "Edge: Range input (O(1) path)" begin
        x_range = 0.0:0.5:4.0
        y_range = collect(10.0:10.0:90.0)
        @test constant_interp(x_range, y_range, 0.25; side = NearestSide()) == 10.0
        @test constant_interp(x_range, y_range, 0.75; side = NearestSide()) == 20.0

        itp_range = constant_interp(x_range, y_range)
        @test itp_range(0.25) == 10.0
    end

end
