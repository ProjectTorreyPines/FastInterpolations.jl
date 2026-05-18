@testitem "_wrap_to_domain — closed boundary (Float64)" begin
    using FastInterpolations: _wrap_to_domain
    # Interior — unchanged
    @test _wrap_to_domain(0.5, 0.0, 1.0) == 0.5
    # Exact right boundary — was wrap, now passthrough
    @test _wrap_to_domain(1.0, 0.0, 1.0) == 1.0
    # Exact left boundary — unchanged
    @test _wrap_to_domain(0.0, 0.0, 1.0) == 0.0
    # Strictly OOB right — still wraps
    @test _wrap_to_domain(1.25, 0.0, 1.0) ≈ 0.25
    # Strictly OOB left — still wraps
    @test _wrap_to_domain(-0.25, 0.0, 1.0) ≈ 0.75
end

@testitem "_wrap_to_domain — closed boundary (generic Real / Int)" begin
    using FastInterpolations: _wrap_to_domain
    # Int xi on Float bounds — exact right boundary
    @test _wrap_to_domain(4, 0.0, 4.0) == 4
    # Float xi on Int-like Float bounds
    @test _wrap_to_domain(4.0, 0.0, 4.0) == 4.0
end

@testitem "_check_domain batch — WrapExtrap returns InBounds at exact last(x)" begin
    using FastInterpolations: _check_domain, WrapExtrap, InBounds
    x = collect(range(0.0, 1.0, 11))
    xq = [0.0, 0.5, 1.0]  # includes both endpoints
    # Closed semantics: all three are in-domain → InBounds()
    @test _check_domain(x, xq, WrapExtrap()) isa InBounds
end

@testitem "_check_domain batch — WrapExtrap returns extrap when OOB" begin
    using FastInterpolations: _check_domain, WrapExtrap
    x = collect(range(0.0, 1.0, 11))
    xq = [0.0, 0.5, 1.25]  # 1.25 is strictly OOB
    @test _check_domain(x, xq, WrapExtrap()) === WrapExtrap()
end
