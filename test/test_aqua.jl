@testitem "Aqua.jl" begin
    using Aqua
    Aqua.test_all(
        FastInterpolations;
        # 55 false positives: Julia's has_unbound_tpars flags TypeVars nested
        # inside Vararg (NTuple{N, AbstractVector{Tg}}, NTuple{N, Tg}, etc.).
        # All are internal _-prefixed functions with no runtime impact.
        unbound_args = (broken = true,),
    )
end
