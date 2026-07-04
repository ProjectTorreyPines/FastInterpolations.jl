# `StorePolicy(copy = false)` must be a true zero-copy contract for narrow-float ND data
# (Float32/Float16/ComplexF32): the stored array IS the caller's array (`itp.data === A`),
# never widened. Pins the fix for the ND builders' grid-eltype-only promotion, which ran
# `Tv.(data)` (an O(n²) Float64 copy) before the store policy was consulted.

@testitem "ND store copy=false is zero-copy for narrow-float data" begin
    FI = FastInterpolations
    build(A) = FI.linear_interp(
        axes(A), A;
        extrap = FI.ClampExtrap(),
        store = FI.StorePolicy(copy = false),
    )

    # The broken set: bare numerics narrower than Float64 (+ Complex), which the
    # legacy promotion widens. `itp.data === A` is the exact zero-copy witness.
    @testset "$T: copy=false aliases the input (no preemptive widening)" for T in
        (Float32, Float16, ComplexF32)
        A = rand(T, 8, 8)
        itp = build(A)
        @test eltype(itp.data) === T   # value array not widened to Float64/ComplexF64
        @test itp.data === A           # stored array IS the caller's array → zero copy
    end

    # Invariant guards — these already alias today; they must not regress under the fix.
    @testset "invariant: already-aliasing types stay aliased" begin
        for A in (rand(Float64, 8, 8), rand(ComplexF64, 8, 8))
            @test build(A).data === A
        end
    end
end

@testitem "ND store copy=false: narrow-float build allocates no O(n²) copy" begin
    FI = FastInterpolations
    _build_alloc(A) = @allocated FI.linear_interp(
        axes(A), A;
        extrap = FI.ClampExtrap(),
        store = FI.StorePolicy(copy = false),
    )
    @testset "$T build is not a full-array copy" for T in (Float32, Float16, ComplexF32)
        A = rand(T, 256, 256)
        _build_alloc(A)                        # warm up (compilation)
        data_bytes = sizeof(T) * length(A)
        # An alias allocates ~0; a widened copy allocates ≥ one data array.
        @test _build_alloc(A) < data_bytes ÷ 4
    end
end
