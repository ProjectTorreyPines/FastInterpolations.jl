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

@testitem "_ExclusivePeriodicData bounds contract (@boundscheck)" begin
    using FastInterpolations: _ExclusivePeriodicData

    # Mirrors `_ExclusivePeriodicAxis` bounds contract — only `i ∈ 1:n+1`
    # is valid, with i==n+1 cyclically returning `inner[1]`.
    n = 4
    y = [10.0, 20.0, 30.0, 40.0]
    c = _ExclusivePeriodicData(y)

    @testset "valid range 1:n+1" begin
        for i in 1:n
            @test c[i] == y[i]
        end
        @test c[n + 1] == y[1]   # cyclic seam
    end

    @testset "out-of-range throws BoundsError" begin
        @test_throws BoundsError c[n + 2]
        @test_throws BoundsError c[100]
        @test_throws BoundsError c[0]
        @test_throws BoundsError c[-3]
    end

    @testset "@inbounds elides the check (no error path taken)" begin
        f(c, i) = @inbounds c[i]
        @test f(c, 1) == 10.0
        @test f(c, n + 1) == 10.0   # cyclic
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

# Lock-down for the data-side `<: AbstractArray{Tv,N}` semantic contract.
# Mirrors `_ExclusivePeriodicAxis` lock-down in test_periodic_axis.jl. Pins
# down: iteration/reduction covers the virtual `(n+1)` span (with the seam
# slot cyclically equal to `inner[1]`), and `similar`/`copy`/broadcast
# materialize the virtual view as a plain `Vector{T}` of length `n+1`.
@testitem "_ExclusivePeriodicData — Base.AbstractArray contract (lock-down)" begin
    using FastInterpolations: _ExclusivePeriodicData

    y = [10.0, 20.0, 30.0, 40.0]
    c = _ExclusivePeriodicData(y)
    n = length(y)

    @testset "Iteration covers virtual span (n+1, includes cyclic seam)" begin
        # `for v in c` and `collect` see the seam at index n+1 with value
        # `inner[1]` (cyclic). INTENDED — eval kernels rely on this uniformity.
        v = collect(c)
        @test length(v) == n + 1
        @test v == [10.0, 20.0, 30.0, 40.0, 10.0]   # seam = inner[1]
    end

    @testset "Reductions span n+1 (seam re-counted as inner[1])" begin
        # `sum(c)` includes the cyclic seam → `sum(inner) + inner[1]`.
        # This is what "sum over a cyclic-extended array" means by definition;
        # diagnostic users should be aware. INTENDED.
        @test sum(c) == sum(y) + y[1]                  # 100 + 10 = 110
        @test maximum(c) == maximum(y)                  # 40
        @test minimum(c) == minimum(y)                  # 10
    end

    @testset "similar/copy materialize as plain Vector (n+1)" begin
        # Default Base path: `similar(::AbstractVector{T}, dims) = Vector{T}(undef, dims)`.
        # Calling `similar(c)` or `copy(c)` produces a NON-cyclic `Vector` of
        # length n+1. Wrapper's zero-copy property is lost — internal callers
        # must use `_raw(c)` or `c.inner` for raw-length buffers.
        s = similar(c)
        @test s isa Vector{Float64}
        @test length(s) == n + 1

        cc = copy(c)
        @test cc isa Vector{Float64}
        @test cc == [10.0, 20.0, 30.0, 40.0, 10.0]     # default copy = collect
    end

    @testset "Broadcast materializes virtual span as plain Vector" begin
        # `c .+ 0` runs through default AbstractArray broadcast → Vector{T} of
        # length n+1. Same escape-hatch rule as `similar`/`copy`: internal
        # wrapper not preserved through broadcast. Do NOT use in hot paths.
        b = c .+ 0.0
        @test b isa Vector{Float64}
        @test length(b) == n + 1
        @test b[n + 1] == y[1]
    end

    @testset "Equality and hash differ from inner" begin
        # Different lengths → NOT equal under `==`. Internal callers that
        # need identity must check `c.inner === reference` directly.
        @test c != c.inner
        @test hash(c) != hash(c.inner)
        # Two wrappers around the same data agree.
        c2 = _ExclusivePeriodicData(y)
        @test c == c2
    end
end
