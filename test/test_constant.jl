# Tests for constant (step/piecewise constant) interpolation

# Import internal items for testing
import FastInterpolations: @_dispatch_side, _constant_kernel, EvalValue, EvalDeriv1, EvalDeriv2

# ============================================================================
# Group 1: Infrastructure Tests (Phase 1)
# ============================================================================
@testset "Constant Interpolation - Infrastructure" begin

    @testset "@_dispatch_side macro" begin
        @testset "dispatch :nearest" begin
            result = @_dispatch_side :nearest => sv begin
                sv
            end
            @test result === Val(:nearest)
        end

        @testset "dispatch :left" begin
            result = @_dispatch_side :left => sv begin
                sv
            end
            @test result === Val(:left)
        end

        @testset "dispatch :right" begin
            result = @_dispatch_side :right => sv begin
                sv
            end
            @test result === Val(:right)
        end

        @testset "dispatch from variable" begin
            for sym in (:nearest, :left, :right)
                result = @_dispatch_side sym => sv begin
                    sv
                end
                @test result === Val(sym)
            end
        end

        @testset "invalid symbol throws ArgumentError" begin
            @test_throws ArgumentError begin
                @_dispatch_side :invalid => sv begin
                    sv
                end
            end
        end

        @testset "body execution with binding" begin
            x = 10
            result = @_dispatch_side :left => sv begin
                (sv, x * 2)
            end
            @test result === (Val(:left), 20)
        end

        @testset "type stability" begin
            function test_dispatch(side::Symbol)
                @_dispatch_side side => sv begin
                    sv
                end
            end
            @test test_dispatch(:nearest) === Val(:nearest)
            @test test_dispatch(:left) === Val(:left)
            @test test_dispatch(:right) === Val(:right)
        end
    end

    @testset "SideVal type" begin
        @testset "SideVal definition" begin
            @test isdefined(FastInterpolations, :SideVal)
            @test FastInterpolations.SideVal isa Union
        end

        @testset "SideVal members" begin
            @test Val(:nearest) isa FastInterpolations.SideVal
            @test Val(:left) isa FastInterpolations.SideVal
            @test Val(:right) isa FastInterpolations.SideVal
        end

        @testset "SideVal exclusions" begin
            @test !(Val(:none) isa FastInterpolations.SideVal)
            @test !(Val(:constant) isa FastInterpolations.SideVal)
            @test !(Val(:other) isa FastInterpolations.SideVal)
        end
    end

end

# ============================================================================
# Group 2: Kernel Tests (Phase 2)
# ============================================================================
@testset "Constant Interpolation - Kernels" begin

    # Test setup: interval [0, 1] with y_left=10.0, y_right=20.0
    y_left = 10.0
    y_right = 20.0
    h = 1.0

    @testset "EvalValue - side=:left" begin
        op = EvalValue()
        sv = Val(:left)
        @test _constant_kernel(op, y_left, y_right, h, 0.0, sv) == 10.0
        @test _constant_kernel(op, y_left, y_right, h, 0.3, sv) == 10.0
        @test _constant_kernel(op, y_left, y_right, h, 0.7, sv) == 10.0
        @test _constant_kernel(op, y_left, y_right, h, 0.99, sv) == 10.0
    end

    @testset "EvalValue - side=:right" begin
        op = EvalValue()
        sv = Val(:right)
        @test _constant_kernel(op, y_left, y_right, h, 0.0, sv) == 10.0  # grid point
        @test _constant_kernel(op, y_left, y_right, h, 0.3, sv) == 20.0
        @test _constant_kernel(op, y_left, y_right, h, 0.7, sv) == 20.0
        @test _constant_kernel(op, y_left, y_right, h, 0.99, sv) == 20.0
    end

    @testset "EvalValue - side=:nearest" begin
        op = EvalValue()
        sv = Val(:nearest)
        @test _constant_kernel(op, y_left, y_right, h, 0.0, sv) == 10.0
        @test _constant_kernel(op, y_left, y_right, h, 0.4, sv) == 10.0
        @test _constant_kernel(op, y_left, y_right, h, 0.5, sv) == 10.0  # tie-breaking
        @test _constant_kernel(op, y_left, y_right, h, 0.6, sv) == 20.0
        @test _constant_kernel(op, y_left, y_right, h, 0.99, sv) == 20.0
    end

    @testset "EvalDeriv1 - all sides return zero" begin
        op = EvalDeriv1()
        @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:nearest)) == 0.0
        @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:left)) == 0.0
        @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:right)) == 0.0
    end

    @testset "EvalDeriv2 - all sides return zero" begin
        op = EvalDeriv2()
        @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:nearest)) == 0.0
        @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:left)) == 0.0
        @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:right)) == 0.0
    end

    @testset "Type preservation" begin
        @test _constant_kernel(EvalValue(), 1.0, 2.0, 1.0, 0.5, Val(:left)) isa Float64
        @test _constant_kernel(EvalDeriv1(), 1.0, 2.0, 1.0, 0.5, Val(:left)) isa Float64
        @test _constant_kernel(EvalValue(), 1.0f0, 2.0f0, 1.0f0, 0.5f0, Val(:nearest)) isa Float32
        @test _constant_kernel(EvalDeriv1(), 1.0f0, 2.0f0, 1.0f0, 0.5f0, Val(:nearest)) isa Float32
    end

    @testset "Non-uniform grid (h != 1)" begin
        op = EvalValue()
        @test _constant_kernel(op, 100.0, 200.0, 2.0, 0.9, Val(:nearest)) == 100.0
        @test _constant_kernel(op, 100.0, 200.0, 2.0, 1.0, Val(:nearest)) == 100.0  # tie
        @test _constant_kernel(op, 100.0, 200.0, 2.0, 1.1, Val(:nearest)) == 200.0
    end

end

# ============================================================================
# Group 3: API Tests (Phase 3)
# ============================================================================
@testset "Constant Interpolation - API" begin

    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]

    @testset "Scalar API" begin
        @test constant_interp(x, y, 0.5) == 10.0
        @test constant_interp(x, y, 1.5) == 20.0
        @test constant_interp(x, y, 2.5) == 30.0

        @test constant_interp(x, y, 0.9; side=:left) == 10.0
        @test constant_interp(x, y, 1.1; side=:left) == 20.0

        @test constant_interp(x, y, 0.0; side=:right) == 10.0
        @test constant_interp(x, y, 0.1; side=:right) == 20.0

        @test constant_interp(x, y, 1.0; side=:nearest) == 20.0
        @test constant_interp(x, y, 1.0; side=:left) == 20.0
        @test constant_interp(x, y, 1.0; side=:right) == 20.0

        @test constant_interp(x, y, 4.0) == 50.0

        @test constant_interp(x, y, 1.5; deriv=1) == 0.0
        @test constant_interp(x, y, 2.5; deriv=2) == 0.0
    end

    @testset "Vector API (allocating)" begin
        result = constant_interp(x, y, [0.5, 1.5, 2.5])
        @test result ≈ [10.0, 20.0, 30.0]

        result_left = constant_interp(x, y, [0.5, 1.5]; side=:left)
        @test result_left ≈ [10.0, 20.0]
    end

    @testset "In-place API" begin
        out = zeros(3)
        constant_interp!(out, x, y, [0.5, 1.5, 2.5])
        @test out ≈ [10.0, 20.0, 30.0]
    end

    @testset "Extrapolation modes" begin
        @test_throws DomainError constant_interp(x, y, -1.0; extrap=:none)
        @test_throws DomainError constant_interp(x, y, 5.0; extrap=:none)

        @test constant_interp(x, y, -1.0; extrap=:constant) == 10.0
        @test constant_interp(x, y, 5.0; extrap=:constant) == 50.0

        @test constant_interp(x, y, -1.0; extrap=:extension) == 10.0
        @test constant_interp(x, y, 5.0; extrap=:extension) == 50.0

        @test constant_interp(x, y, 4.0; extrap=:wrap) == 10.0
        @test constant_interp(x, y, 4.5; extrap=:wrap) == 10.0

        @test constant_interp(x, y, -1.0; extrap=:constant, deriv=1) == 0.0
    end

    @testset "Real type wrapper (Integer input)" begin
        x_int = [0, 1, 2, 3, 4]
        y_int = [10, 20, 30, 40, 50]
        result = constant_interp(x_int, y_int, 1.5)
        @test result isa Float64
        @test result == 20.0
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
        @test itp(1.5; deriv=1) == 0.0
        @test itp(2.5; deriv=2) == 0.0
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
        itp_wrap = constant_interp(x, y; extrap=:wrap)
        @test itp_wrap(4.0) == 10.0

        itp_const = constant_interp(x, y; extrap=:constant)
        @test itp_const(-1.0) == 10.0
        @test itp_const(5.0) == 50.0

        itp_left = constant_interp(x, y; side=:left)
        @test itp_left(0.9) == 10.0

        itp_right = constant_interp(x, y; side=:right)
        @test itp_right(0.1) == 20.0
    end

    @testset "Type inference" begin
        itp = constant_interp(x, y)
        @test @inferred(itp(0.5)) isa Float64
        @test @inferred(constant_interp(x, y, 0.5)) isa Float64
        @test @inferred(constant_interp(x, y, 0.5; side=:left)) isa Float64
        @test @inferred(constant_interp(x, y, 0.5; extrap=:constant)) isa Float64
        @test @inferred(constant_interp(x, y, 0.5; deriv=1)) isa Float64
    end

end

# ============================================================================
# Group 4: Allocation Tests (Phase 3)
# ============================================================================
@testset "Constant Interpolation - Allocations" begin

    x = collect(0.0:0.1:1.0)
    y = sin.(x)

    @testset "constant_interp! in-place" begin
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        constant_interp!(out, x, y, xq)
        allocs = @allocated constant_interp!(out, x, y, xq)
        @test allocs == 0
    end

    @testset "constant_interp! with side option" begin
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        constant_interp!(out, x, y, xq; side=:left)
        allocs = @allocated constant_interp!(out, x, y, xq; side=:left)
        @test allocs == 0
    end

    @testset "constant_interp! with extrap option" begin
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        constant_interp!(out, x, y, xq; extrap=:constant)
        allocs = @allocated constant_interp!(out, x, y, xq; extrap=:constant)
        @test allocs == 0
    end

    @testset "ConstantInterpolant scalar" begin
        itp = constant_interp(x, y)
        itp(0.5)
        allocs = @allocated itp(0.55)
        @test allocs == 0
    end

    @testset "ConstantInterpolant scalar (legacy threshold)" begin
        x5 = [0.0, 1.0, 2.0, 3.0, 4.0]
        y5 = [10.0, 20.0, 30.0, 40.0, 50.0]
        itp = constant_interp(x5, y5)
        itp(0.5)
        allocs = @allocated itp(0.55)
        @test allocs <= 256  # Allow small allocation on older Julia
    end

    @testset "ConstantInterpolant in-place vector" begin
        itp = constant_interp(x, y)
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        itp(out, xq)
        allocs = @allocated itp(out, xq)
        @test allocs == 0
    end

    @testset "ConstantInterpolant in-place with deriv" begin
        itp = constant_interp(x, y)
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        itp(out, xq; deriv=1)
        allocs = @allocated itp(out, xq; deriv=1)
        @test allocs == 0
    end

    @testset "ConstantInterpolant with extrap=:wrap in-place" begin
        itp = constant_interp(x, y; extrap=:wrap)
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        itp(out, xq)
        allocs = @allocated itp(out, xq)
        @test allocs == 0
    end

    @testset "ConstantInterpolant with side=:left in-place" begin
        itp = constant_interp(x, y; side=:left)
        out = zeros(5)
        xq = [0.1, 0.3, 0.5, 0.7, 0.9]
        itp(out, xq)
        allocs = @allocated itp(out, xq)
        @test allocs == 0
    end

end

# ============================================================================
# Group 5: Integration & Edge Cases (Phase 4)
# ============================================================================
@testset "Constant Interpolation - Integration" begin

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
        itp_const = constant_interp(x, y; extrap=:constant)
        d1 = deriv1(itp_const)
        @test d1(-1.0) == 0.0
        @test d1(5.0) == 0.0

        itp_wrap = constant_interp(x, y; extrap=:wrap)
        d2 = deriv2(itp_wrap)
        @test d2(4.5) == 0.0

        itp_left = constant_interp(x, y; side=:left)
        @test deriv1(itp_left)(0.5) == 0.0

        itp_right = constant_interp(x, y; side=:right)
        @test deriv2(itp_right)(0.5) == 0.0
    end

    @testset "Edge: Grid point behavior" begin
        for side in (:nearest, :left, :right)
            @test constant_interp(x, y, 0.0; side=side) == 10.0
            @test constant_interp(x, y, 1.0; side=side) == 20.0
            @test constant_interp(x, y, 2.0; side=side) == 30.0
            @test constant_interp(x, y, 3.0; side=side) == 40.0
            @test constant_interp(x, y, 4.0; side=side) == 50.0
        end
    end

    @testset "Edge: xi == x[end] with extrap modes" begin
        @test constant_interp(x, y, 4.0; extrap=:none) == 50.0
        @test constant_interp(x, y, 4.0; extrap=:constant) == 50.0
        @test constant_interp(x, y, 4.0; extrap=:extension) == 50.0
        @test constant_interp(x, y, 4.0; extrap=:wrap) == 10.0
    end

    @testset "Edge: Wrap boundary cases" begin
        @test constant_interp(x, y, 4.0; extrap=:wrap, side=:left) == 10.0
        @test constant_interp(x, y, 4.0; extrap=:wrap, side=:right) == 10.0
        @test constant_interp(x, y, 4.0; extrap=:wrap, side=:nearest) == 10.0
        @test constant_interp(x, y, 4.5; extrap=:wrap, side=:nearest) == 10.0
        @test constant_interp(x, y, 5.0; extrap=:wrap, side=:nearest) == 20.0
    end

    @testset "Edge: Extrapolation + deriv" begin
        @test constant_interp(x, y, -1.0; extrap=:constant, deriv=1) == 0.0
        @test constant_interp(x, y, 5.0; extrap=:constant, deriv=2) == 0.0
        @test constant_interp(x, y, -1.0; extrap=:extension, deriv=1) == 0.0
        @test constant_interp(x, y, 5.0; extrap=:extension, deriv=2) == 0.0
        @test constant_interp(x, y, 4.5; extrap=:wrap, deriv=1) == 0.0
    end

    @testset "Edge: Midpoint tie-breaking" begin
        @test constant_interp(x, y, 0.5; side=:nearest) == 10.0
        @test constant_interp(x, y, 1.5; side=:nearest) == 20.0
        @test constant_interp(x, y, 2.5; side=:nearest) == 30.0
    end

    @testset "Edge: Non-uniform grid" begin
        x_nu = [0.0, 0.5, 2.0, 2.5, 4.0]
        y_nu = [10.0, 20.0, 30.0, 40.0, 50.0]
        @test constant_interp(x_nu, y_nu, 0.25; side=:nearest) == 10.0
        @test constant_interp(x_nu, y_nu, 1.0; side=:nearest) == 20.0
        @test constant_interp(x_nu, y_nu, 1.5; side=:nearest) == 30.0
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
        @test constant_interp(x_min, y_min, 0.5; side=:nearest) == 10.0
        @test constant_interp(x_min, y_min, 0.5; side=:left) == 10.0
        @test constant_interp(x_min, y_min, 0.5; side=:right) == 20.0
        @test constant_interp(x_min, y_min, 0.0) == 10.0
        @test constant_interp(x_min, y_min, 1.0) == 20.0
    end

    @testset "Edge: Range input (O(1) path)" begin
        x_range = 0.0:0.5:4.0
        y_range = collect(10.0:10.0:90.0)
        @test constant_interp(x_range, y_range, 0.25; side=:nearest) == 10.0
        @test constant_interp(x_range, y_range, 0.75; side=:nearest) == 20.0

        itp_range = constant_interp(x_range, y_range)
        @test itp_range(0.25) == 10.0
    end

end
