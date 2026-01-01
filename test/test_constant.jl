# Tests for constant (step/piecewise constant) interpolation
# Phase 1: Infrastructure tests - @_dispatch_side macro and SideVal type

# Import internal macro for testing
import FastInterpolations: @_dispatch_side

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

end
