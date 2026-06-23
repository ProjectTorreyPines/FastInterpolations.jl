# Type-promotion contract for the integrate path: the `_integrate_op` witness floats
# Int, lifts Dual, and stays duck-safe; integrate over Int/Dual grids returns the
# right element type. Phase 1 of the integrate promotion modernization.

@testitem "_integrate_op witness eltype matrix" begin
    using FastInterpolations: _integrate_op, _promote_eltype
    using ForwardDiff: Dual
    D = Dual{Nothing, Float64, 1}
    # (Tg, Tv, Tspan) → Tout
    @test _promote_eltype(_integrate_op, Float64, Float64, Float64) === Float64
    @test _promote_eltype(_integrate_op, Int, Int, Int) === Float64          # floats Int
    @test _promote_eltype(_integrate_op, D, Float64, Float64) === D          # AD wrt grid
    @test _promote_eltype(_integrate_op, Float64, Float64, D) === D          # AD wrt bounds
    @test _promote_eltype(_integrate_op, Float64, Int, Float64) === Float64  # Int data floats
end
