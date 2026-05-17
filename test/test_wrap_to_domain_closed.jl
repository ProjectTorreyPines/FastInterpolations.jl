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
