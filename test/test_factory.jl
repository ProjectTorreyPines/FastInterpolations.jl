using Test
using FastInterpolations
using FastInterpolations: AbstractSearchPolicy, AbstractExtrap, AbstractSide

@testset "Factory Functions" begin

    # ========================================
    # Search Factory
    # ========================================

    @testset "Search" begin
        @testset "Symbol → Concrete Type" begin
            @test Search(:auto) isa AutoSearch
            @test Search(:binary) isa BinarySearch
            @test Search(:linear) isa LinearSearch
            @test Search(:linear_binary) isa LinearBinarySearch{8}  # default window
        end

        @testset "LinearBinarySearch kwargs forwarding" begin
            @test Search(:linear_binary; linear_window=4) isa LinearBinarySearch{4}
            @test Search(:linear_binary; linear_window=2) isa LinearBinarySearch{2}
            @test Search(:linear_binary; linear_window=16) isa LinearBinarySearch{16}
            @test Search(:linear_binary; linear_window=0) isa LinearBinarySearch{0}
            @test Search(:linear_binary; linear_window=128) isa LinearBinarySearch{128}
        end

        @testset "Passthrough" begin
            @test Search(AutoSearch()) === AutoSearch()
            @test Search(BinarySearch()) === BinarySearch()
            @test Search(LinearSearch()) === LinearSearch()
            lbs = LinearBinarySearch{4}()
            @test Search(lbs) === lbs
        end

        @testset "Error handling" begin
            # Unknown symbol
            @test_throws ArgumentError Search(:unknown)
            @test_throws ArgumentError Search(:bianry)  # typo

            # Invalid kwargs for non-:linear_binary
            @test_throws ArgumentError Search(:binary; linear_window=8)
            @test_throws ArgumentError Search(:auto; linear_window=4)
            @test_throws ArgumentError Search(:linear; linear_window=2)

            # Invalid linear_window value (forwarded to LinearBinarySearch)
            @test_throws ArgumentError Search(:linear_binary; linear_window=3)
            @test_throws ArgumentError Search(:linear_binary; linear_window=256)
        end

        @testset "Error messages are informative" begin
            try
                Search(:bianry)
            catch e
                @test e isa ArgumentError
                @test occursin("valid options", e.msg)
                @test occursin(":binary", e.msg)
            end

            try
                Search(:binary; linear_window=8)
            catch e
                @test e isa ArgumentError
                @test occursin(":linear_binary", e.msg)
            end
        end
    end

    # ========================================
    # Extrap Factory
    # ========================================

    @testset "Extrap" begin
        @testset "Symbol → Concrete Type" begin
            @test Extrap(:none) isa NoExtrap
            @test Extrap(:constant) isa ClampedExtrap
            @test Extrap(:extend) isa ExtendExtrap
            @test Extrap(:wrap) isa WrapExtrap
        end

        @testset "Passthrough" begin
            @test Extrap(NoExtrap()) === NoExtrap()
            @test Extrap(ClampedExtrap()) === ClampedExtrap()
            @test Extrap(ExtendExtrap()) === ExtendExtrap()
            @test Extrap(WrapExtrap()) === WrapExtrap()
        end

        @testset "Error handling" begin
            @test_throws ArgumentError Extrap(:unknown)
            @test_throws ArgumentError Extrap(:no)        # must be :none
            @test_throws ArgumentError Extrap(:const)      # must be :constant
            @test_throws ArgumentError Extrap(:periodic)   # not an extrap mode
        end

        @testset "Error messages are informative" begin
            try
                Extrap(:no)
            catch e
                @test e isa ArgumentError
                @test occursin("valid options", e.msg)
                @test occursin(":none", e.msg)
            end
        end
    end

    # ========================================
    # Side Factory
    # ========================================

    @testset "Side" begin
        @testset "Symbol → Concrete Type" begin
            @test Side(:nearest) isa NearestSide
            @test Side(:left) isa LeftSide
            @test Side(:right) isa RightSide
        end

        @testset "Passthrough" begin
            @test Side(NearestSide()) === NearestSide()
            @test Side(LeftSide()) === LeftSide()
            @test Side(RightSide()) === RightSide()
        end

        @testset "Error handling" begin
            @test_throws ArgumentError Side(:unknown)
            @test_throws ArgumentError Side(:center)
        end

        @testset "Error messages are informative" begin
            try
                Side(:center)
            catch e
                @test e isa ArgumentError
                @test occursin("valid options", e.msg)
                @test occursin(":left", e.msg)
            end
        end
    end

    # ========================================
    # ND Variadic (Multi-arg → Tuple) Tests
    # ========================================

    @testset "ND Variadic Forms" begin
        @testset "Search variadic" begin
            result = Search(:binary, :linear_binary)
            @test result isa Tuple{BinarySearch, LinearBinarySearch{8}}
            @test result == (BinarySearch(), LinearBinarySearch{8}())

            result3 = Search(:binary, :auto, :linear)
            @test result3 isa Tuple{BinarySearch, AutoSearch, LinearSearch}
            @test length(result3) == 3
        end

        @testset "Extrap variadic" begin
            result = Extrap(:extend, :none, :wrap)
            @test result isa Tuple{ExtendExtrap, NoExtrap, WrapExtrap}
            @test result == (ExtendExtrap(), NoExtrap(), WrapExtrap())

            result2 = Extrap(:constant, :extend)
            @test result2 isa Tuple{ClampedExtrap, ExtendExtrap}
        end

        @testset "Side variadic" begin
            result = Side(:left, :nearest)
            @test result isa Tuple{LeftSide, NearestSide}
            @test result == (LeftSide(), NearestSide())

            result3 = Side(:left, :right, :nearest)
            @test result3 isa Tuple{LeftSide, RightSide, NearestSide}
        end

        @testset "Variadic error propagation" begin
            @test_throws ArgumentError Extrap(:extend, :invalid, :wrap)
            @test_throws ArgumentError Search(:binary, :unknown)
            @test_throws ArgumentError Side(:left, :center)
        end
    end

    # ========================================
    # Zero-Allocation Tests
    # ========================================

    @testset "Zero Allocation" begin
        function _test_search_alloc()
            _ = Search(:binary)
            _ = Search(:auto)
            _ = Search(:linear_binary)
            _ = Search(:binary, :auto)
            a1 = @allocated Search(:binary)
            a2 = @allocated Search(:auto)
            a3 = @allocated Search(:linear)
            a4 = @allocated Search(:linear_binary)
            a5 = @allocated Search(BinarySearch())
            a6 = @allocated Search(:binary, :auto, :linear)
            return (a1, a2, a3, a4, a5, a6)
        end

        function _test_extrap_alloc()
            _ = Extrap(:none)
            _ = Extrap(:constant)
            _ = Extrap(:extend, :none)
            a1 = @allocated Extrap(:none)
            a2 = @allocated Extrap(:constant)
            a3 = @allocated Extrap(:extend)
            a4 = @allocated Extrap(:wrap)
            a5 = @allocated Extrap(NoExtrap())
            a6 = @allocated Extrap(:extend, :none, :wrap)
            return (a1, a2, a3, a4, a5, a6)
        end

        function _test_side_alloc()
            _ = Side(:nearest)
            _ = Side(:left)
            _ = Side(:left, :right)
            a1 = @allocated Side(:nearest)
            a2 = @allocated Side(:left)
            a3 = @allocated Side(:right)
            a4 = @allocated Side(NearestSide())
            a5 = @allocated Side(:left, :nearest)
            return (a1, a2, a3, a4, a5)
        end

        @testset "Search zero-alloc" begin
            allocs = _test_search_alloc()
            for (i, a) in enumerate(allocs)
                @test a == 0
            end
        end

        @testset "Extrap zero-alloc" begin
            allocs = _test_extrap_alloc()
            for (i, a) in enumerate(allocs)
                @test a == 0
            end
        end

        @testset "Side zero-alloc" begin
            allocs = _test_side_alloc()
            for (i, a) in enumerate(allocs)
                @test a == 0
            end
        end
    end

    # ========================================
    # End-to-End Integration
    # ========================================

    @testset "End-to-End Integration" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(x)

        @testset "Factories work in interpolant construction" begin
            itp_s = cubic_interp(x, y; search=Search(:binary), extrap=Extrap(:extend))
            @test itp_s(0.5) isa Float64

            itp_e = linear_interp(x, y; extrap=Extrap(:constant))
            @test itp_e(-0.1) == itp_e(0.0)  # clamped

            itp_c = constant_interp(x, y; side=Side(:left))
            @test itp_c(0.05) isa Float64
        end

        @testset "Factories produce identical results to direct types" begin
            itp_factory = cubic_interp(x, y; search=Search(:binary), extrap=Extrap(:extend))
            itp_direct  = cubic_interp(x, y; search=BinarySearch(), extrap=ExtendExtrap())

            for xq in [0.0, 0.25, 0.5, 0.75, 1.0, 1.1, -0.1]
                @test itp_factory(xq) == itp_direct(xq)
            end
        end
    end

end
