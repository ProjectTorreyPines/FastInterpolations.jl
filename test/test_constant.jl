# Tests for constant (step/piecewise constant) interpolation

# Import internal items for testing
import FastInterpolations: @_dispatch_side, _constant_kernel, EvalValue, EvalDeriv1, EvalDeriv2

@testset "Constant Interpolation" begin

    # ========================================
    # Phase 1: @_dispatch_side Macro Tests
    # ========================================
    @testset "@_dispatch_side macro" begin

        # Test 1: Basic dispatch - :nearest
        @testset "dispatch :nearest" begin
            result = @_dispatch_side :nearest => sv begin
                sv
            end
            @test result === Val(:nearest)
        end

        # Test 2: Basic dispatch - :left
        @testset "dispatch :left" begin
            result = @_dispatch_side :left => sv begin
                sv
            end
            @test result === Val(:left)
        end

        # Test 3: Basic dispatch - :right
        @testset "dispatch :right" begin
            result = @_dispatch_side :right => sv begin
                sv
            end
            @test result === Val(:right)
        end

        # Test 4: Dynamic dispatch from variable
        @testset "dispatch from variable" begin
            for sym in (:nearest, :left, :right)
                result = @_dispatch_side sym => sv begin
                    sv
                end
                @test result === Val(sym)
            end
        end

        # Test 5: Invalid symbol throws ArgumentError
        @testset "invalid symbol throws ArgumentError" begin
            @test_throws ArgumentError begin
                @_dispatch_side :invalid => sv begin
                    sv
                end
            end
        end

        # Test 6: Body expression is executed with correct binding
        @testset "body execution with binding" begin
            x = 10
            result = @_dispatch_side :left => sv begin
                (sv, x * 2)
            end
            @test result === (Val(:left), 20)
        end

        # Test 7: Type stability - result should be inferred
        @testset "type stability" begin
            function test_dispatch(side::Symbol)
                @_dispatch_side side => sv begin
                    sv
                end
            end
            # SideVal is Union{Val{:nearest}, Val{:left}, Val{:right}}
            # Julia should infer Union correctly
            @test test_dispatch(:nearest) === Val(:nearest)
            @test test_dispatch(:left) === Val(:left)
            @test test_dispatch(:right) === Val(:right)
        end

    end

    # ========================================
    # Phase 1: SideVal Type Tests
    # ========================================
    @testset "SideVal type" begin

        # Test 1: SideVal is defined and is a Union
        @testset "SideVal definition" begin
            @test isdefined(FastInterpolations, :SideVal)
            @test FastInterpolations.SideVal isa Union
        end

        # Test 2: SideVal includes all three Val types
        @testset "SideVal members" begin
            @test Val(:nearest) isa FastInterpolations.SideVal
            @test Val(:left) isa FastInterpolations.SideVal
            @test Val(:right) isa FastInterpolations.SideVal
        end

        # Test 3: Other Val types are NOT SideVal
        @testset "SideVal exclusions" begin
            @test !(Val(:none) isa FastInterpolations.SideVal)
            @test !(Val(:constant) isa FastInterpolations.SideVal)
            @test !(Val(:other) isa FastInterpolations.SideVal)
        end

    end

    # ========================================
    # Phase 2: _constant_kernel Tests
    # ========================================
    @testset "_constant_kernel" begin

        # Test setup: interval [0, 1] with y_left=10.0, y_right=20.0
        # h = 1.0, dt1 varies
        y_left = 10.0
        y_right = 20.0
        h = 1.0

        @testset "EvalValue - side=:left" begin
            op = EvalValue()
            sv = Val(:left)
            # :left always returns y_left regardless of position
            @test _constant_kernel(op, y_left, y_right, h, 0.0, sv) == 10.0  # at left boundary
            @test _constant_kernel(op, y_left, y_right, h, 0.3, sv) == 10.0  # 30% into interval
            @test _constant_kernel(op, y_left, y_right, h, 0.7, sv) == 10.0  # 70% into interval
            @test _constant_kernel(op, y_left, y_right, h, 0.99, sv) == 10.0 # near right boundary
        end

        @testset "EvalValue - side=:right" begin
            op = EvalValue()
            sv = Val(:right)
            # :right returns y_left at grid point (dt1==0), y_right otherwise
            @test _constant_kernel(op, y_left, y_right, h, 0.0, sv) == 10.0  # grid point → y_left
            @test _constant_kernel(op, y_left, y_right, h, 0.3, sv) == 20.0  # off grid → y_right
            @test _constant_kernel(op, y_left, y_right, h, 0.7, sv) == 20.0  # off grid → y_right
            @test _constant_kernel(op, y_left, y_right, h, 0.99, sv) == 20.0 # off grid → y_right
        end

        @testset "EvalValue - side=:nearest" begin
            op = EvalValue()
            sv = Val(:nearest)
            # :nearest returns y_left if dt1 <= h/2, y_right otherwise (left tie-breaking)
            @test _constant_kernel(op, y_left, y_right, h, 0.0, sv) == 10.0  # at left → y_left
            @test _constant_kernel(op, y_left, y_right, h, 0.4, sv) == 10.0  # < midpoint → y_left
            @test _constant_kernel(op, y_left, y_right, h, 0.5, sv) == 10.0  # midpoint → y_left (tie-breaking)
            @test _constant_kernel(op, y_left, y_right, h, 0.6, sv) == 20.0  # > midpoint → y_right
            @test _constant_kernel(op, y_left, y_right, h, 0.99, sv) == 20.0 # near right → y_right
        end

        @testset "EvalDeriv1 - all sides return zero" begin
            op = EvalDeriv1()
            # Constant function has zero derivative everywhere
            @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:nearest)) == 0.0
            @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:left)) == 0.0
            @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:right)) == 0.0
        end

        @testset "EvalDeriv2 - all sides return zero" begin
            op = EvalDeriv2()
            # Constant function has zero second derivative everywhere
            @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:nearest)) == 0.0
            @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:left)) == 0.0
            @test _constant_kernel(op, y_left, y_right, h, 0.5, Val(:right)) == 0.0
        end

        @testset "Type preservation" begin
            # Float64
            @test _constant_kernel(EvalValue(), 1.0, 2.0, 1.0, 0.5, Val(:left)) isa Float64
            @test _constant_kernel(EvalDeriv1(), 1.0, 2.0, 1.0, 0.5, Val(:left)) isa Float64
            # Float32
            @test _constant_kernel(EvalValue(), 1.0f0, 2.0f0, 1.0f0, 0.5f0, Val(:nearest)) isa Float32
            @test _constant_kernel(EvalDeriv1(), 1.0f0, 2.0f0, 1.0f0, 0.5f0, Val(:nearest)) isa Float32
        end

        @testset "Non-uniform grid (h != 1)" begin
            op = EvalValue()
            # Interval [0, 2] → h=2.0, midpoint at dt1=1.0
            @test _constant_kernel(op, 100.0, 200.0, 2.0, 0.9, Val(:nearest)) == 100.0  # < midpoint
            @test _constant_kernel(op, 100.0, 200.0, 2.0, 1.0, Val(:nearest)) == 100.0  # = midpoint (tie)
            @test _constant_kernel(op, 100.0, 200.0, 2.0, 1.1, Val(:nearest)) == 200.0  # > midpoint
        end

    end

    # ========================================
    # Phase 3: constant_interp API Tests
    # ========================================
    @testset "constant_interp API" begin

        # Test data: step function
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]

        @testset "Scalar API" begin
            # Basic interpolation with default :nearest
            @test constant_interp(x, y, 0.5) == 10.0  # < midpoint → left
            @test constant_interp(x, y, 1.5) == 20.0  # at 1.0 interval
            @test constant_interp(x, y, 2.5) == 30.0

            # side=:left
            @test constant_interp(x, y, 0.9; side=:left) == 10.0
            @test constant_interp(x, y, 1.1; side=:left) == 20.0

            # side=:right
            @test constant_interp(x, y, 0.0; side=:right) == 10.0  # grid point
            @test constant_interp(x, y, 0.1; side=:right) == 20.0  # off grid

            # Grid points - all sides return y[i]
            @test constant_interp(x, y, 1.0; side=:nearest) == 20.0
            @test constant_interp(x, y, 1.0; side=:left) == 20.0
            @test constant_interp(x, y, 1.0; side=:right) == 20.0

            # Boundary: x[end]
            @test constant_interp(x, y, 4.0) == 50.0

            # Derivatives always zero
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
            # :none throws DomainError
            @test_throws DomainError constant_interp(x, y, -1.0; extrap=:none)
            @test_throws DomainError constant_interp(x, y, 5.0; extrap=:none)

            # :constant clamps to boundary (side ignored in extrap)
            @test constant_interp(x, y, -1.0; extrap=:constant) == 10.0
            @test constant_interp(x, y, 5.0; extrap=:constant) == 50.0

            # :extension same as :constant for step function
            @test constant_interp(x, y, -1.0; extrap=:extension) == 10.0
            @test constant_interp(x, y, 5.0; extrap=:extension) == 50.0

            # :wrap - half-open interval [x_min, x_max)
            @test constant_interp(x, y, 4.0; extrap=:wrap) == 10.0  # x_max → x_min
            @test constant_interp(x, y, 4.5; extrap=:wrap) == 10.0  # wraps to 0.5

            # Extrap + deriv returns zero
            @test constant_interp(x, y, -1.0; extrap=:constant, deriv=1) == 0.0
        end

        @testset "Real type wrapper (Integer input)" begin
            x_int = [0, 1, 2, 3, 4]
            y_int = [10, 20, 30, 40, 50]
            # Should auto-promote to Float64
            result = constant_interp(x_int, y_int, 1.5)
            @test result isa Float64
            @test result == 20.0
        end

    end

    # ========================================
    # Phase 3: ConstantInterpolant Tests
    # ========================================
    @testset "ConstantInterpolant" begin

        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]

        @testset "2-arg form returns interpolant" begin
            itp = constant_interp(x, y)
            @test itp isa FastInterpolations.ConstantInterpolant
        end

        @testset "Scalar call" begin
            itp = constant_interp(x, y)
            @test itp(0.5) == 10.0
            @test itp(1.5) == 20.0
            @test itp(4.0) == 50.0

            # With deriv keyword
            @test itp(1.5; deriv=1) == 0.0
            @test itp(2.5; deriv=2) == 0.0
        end

        @testset "Vector call" begin
            itp = constant_interp(x, y)
            result = itp([0.5, 1.5, 2.5])
            @test result ≈ [10.0, 20.0, 30.0]
        end

        @testset "Broadcast fusion" begin
            itp = constant_interp(x, y)
            # Broadcasting
            result = itp.([0.5, 1.5])
            @test result ≈ [10.0, 20.0]

            # Fused broadcast
            coef = 2.0
            query = [0.5, 1.5]
            result_fused = @. coef * itp(query)
            @test result_fused ≈ [20.0, 40.0]
        end

        @testset "Extrapolation options" begin
            itp_wrap = constant_interp(x, y; extrap=:wrap)
            @test itp_wrap(4.0) == 10.0  # wraps

            itp_const = constant_interp(x, y; extrap=:constant)
            @test itp_const(-1.0) == 10.0
            @test itp_const(5.0) == 50.0
        end

        @testset "Side options" begin
            itp_left = constant_interp(x, y; side=:left)
            @test itp_left(0.9) == 10.0

            itp_right = constant_interp(x, y; side=:right)
            @test itp_right(0.1) == 20.0
        end

        @testset "Type inference" begin
            itp = constant_interp(x, y)
            @test @inferred(itp(0.5)) isa Float64
        end

        @testset "Zero allocation (scalar)" begin
            itp = constant_interp(x, y)
            itp(0.5)  # warmup
            allocs = @allocated itp(0.55)
            # Allow small allocation on older Julia
            @test allocs <= 256
        end

        @testset "Zero allocation (in-place vector)" begin
            itp = constant_interp(x, y)
            out = zeros(3)
            xq = [0.5, 1.5, 2.5]

            # Warmup
            itp(out, xq)

            # Test zero allocation
            allocs = @allocated itp(out, xq)
            @test allocs == 0
        end

    end

    # ========================================
    # Phase 3: Zero Allocation Tests
    # ========================================
    @testset "Zero Allocation Verification" begin

        x = collect(0.0:0.1:1.0)
        y = sin.(x)

        @testset "constant_interp! in-place" begin
            out = zeros(5)
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]

            # Warmup
            constant_interp!(out, x, y, xq)

            # Test zero allocation
            allocs = @allocated constant_interp!(out, x, y, xq)
            @test allocs == 0
        end

        @testset "constant_interp! with side option" begin
            out = zeros(5)
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]

            # Warmup
            constant_interp!(out, x, y, xq; side=:left)

            # Test zero allocation
            allocs = @allocated constant_interp!(out, x, y, xq; side=:left)
            @test allocs == 0
        end

        @testset "constant_interp! with extrap option" begin
            out = zeros(5)
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]

            # Warmup
            constant_interp!(out, x, y, xq; extrap=:constant)

            # Test zero allocation
            allocs = @allocated constant_interp!(out, x, y, xq; extrap=:constant)
            @test allocs == 0
        end

        @testset "ConstantInterpolant scalar" begin
            itp = constant_interp(x, y)

            # Warmup
            itp(0.5)

            # Test zero allocation
            allocs = @allocated itp(0.55)
            @test allocs == 0
        end

        @testset "ConstantInterpolant in-place vector" begin
            itp = constant_interp(x, y)
            out = zeros(5)
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]

            # Warmup
            itp(out, xq)

            # Test zero allocation
            allocs = @allocated itp(out, xq)
            @test allocs == 0
        end

        @testset "ConstantInterpolant in-place with deriv" begin
            itp = constant_interp(x, y)
            out = zeros(5)
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]

            # Warmup
            itp(out, xq; deriv=1)

            # Test zero allocation
            allocs = @allocated itp(out, xq; deriv=1)
            @test allocs == 0
        end

        @testset "ConstantInterpolant with extrap=:wrap in-place" begin
            itp = constant_interp(x, y; extrap=:wrap)
            out = zeros(5)
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]

            # Warmup
            itp(out, xq)

            # Test zero allocation
            allocs = @allocated itp(out, xq)
            @test allocs == 0
        end

        @testset "ConstantInterpolant with side=:left in-place" begin
            itp = constant_interp(x, y; side=:left)
            out = zeros(5)
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]

            # Warmup
            itp(out, xq)

            # Test zero allocation
            allocs = @allocated itp(out, xq)
            @test allocs == 0
        end

    end

end
