@testitem "_ExclusivePeriodicData construction + interface (1D)" begin
    using FastInterpolations: _ExclusivePeriodicData

    @testset "Float64 vector inner — basic round-trip" begin
        y = [10.0, 20.0, 30.0, 40.0]
        c = _ExclusivePeriodicData(y)

        @test c isa _ExclusivePeriodicData{Float64, 1, Vector{Float64}}
        @test c.inner === y

        # length reports virtual extension (n+1)
        @test length(c) == 5
        @test size(c) == (5,)
        @test firstindex(c) == 1
        @test lastindex(c) == 5
        @test eltype(c) == Float64

        # getindex: forwards to inner for i ≤ n, cyclic at i = n+1
        for i in 1:4
            @test c[i] == y[i]
        end
        @test c[5] == y[1]   # cyclic — wrapped tail returns y[1]
    end

    @testset "first / last (cyclic boundary)" begin
        y = [10.0, 20.0, 30.0, 40.0]
        c = _ExclusivePeriodicData(y)

        @test first(c) == y[1] == 10.0
        # `last` is *also* y[1] because the wrapped tail equals the wrapped head
        @test last(c) == y[1] == 10.0
    end

    @testset "Float32 + Int + Complex" begin
        y_f32 = Float32[1.0, 2.0, 3.0]
        c_f32 = _ExclusivePeriodicData(y_f32)
        @test c_f32 isa _ExclusivePeriodicData{Float32, 1}
        @test eltype(c_f32) == Float32
        @test c_f32[4] === Float32(1.0)

        y_int = [1, 2, 3, 4]
        c_int = _ExclusivePeriodicData(y_int)
        @test c_int isa _ExclusivePeriodicData{Int, 1}
        @test c_int[5] === 1

        y_c = [1.0 + 2im, 3.0 + 4im, 5.0 + 6im]
        c_c = _ExclusivePeriodicData(y_c)
        @test c_c isa _ExclusivePeriodicData{ComplexF64, 1}
        @test c_c[4] === 1.0 + 2im
    end

    @testset "Idempotent: re-wrap returns input unchanged" begin
        y = [1.0, 2.0, 3.0]
        c1 = _ExclusivePeriodicData(y)
        c2 = _ExclusivePeriodicData(c1)
        @test c2 === c1
    end

    @testset "AbstractArray interface" begin
        y = [10.0, 20.0, 30.0]
        c = _ExclusivePeriodicData(y)

        # `<: AbstractArray{Tv, 1}` so most Base ops work
        @test c isa AbstractArray{Float64, 1}
        @test c isa AbstractVector{Float64}

        # Iteration covers all virtual slots
        @test collect(c) == [10.0, 20.0, 30.0, 10.0]   # last is cyclic

        # sum/maximum/etc. iterate the virtual extension too
        @test sum(c) == 70.0   # 10+20+30+10
        @test maximum(c) == 30.0
    end
end

@testitem "_ExclusivePeriodicData + _ExclusivePeriodicAxis pairing (length compatibility)" begin
    using FastInterpolations: _ExclusivePeriodicData, _ExclusivePeriodicAxis,
                              _check_compatible_length

    @testset "Wrapped pair: length(x) == length(y) (both n+1)" begin
        x = [0.0, 0.25, 0.5, 0.75]
        y = [10.0, 20.0, 30.0, 40.0]
        xa = _ExclusivePeriodicAxis(x, 1.0)
        yd = _ExclusivePeriodicData(y)

        @test length(xa) == 5
        @test length(yd) == 5

        # Single generic `_check_compatible_length(x, y)` works uniformly
        # because both wrappers report n+1.
        @test _check_compatible_length(xa, yd) === nothing
    end

    @testset "Plain vectors still validated" begin
        @test _check_compatible_length([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) === nothing
        @test_throws ArgumentError _check_compatible_length([1.0, 2.0], [10.0, 20.0, 30.0])
    end

    @testset "Mixed wrapper / raw rejects mismatch" begin
        x = [0.0, 0.25, 0.5]
        xa = _ExclusivePeriodicAxis(x, 1.0)        # length 4 (virtual)
        y_raw = [10.0, 20.0, 30.0]                  # length 3
        # length(xa) = 4 ≠ length(y_raw) = 3 — mismatch
        @test_throws ArgumentError _check_compatible_length(xa, y_raw)
    end
end
