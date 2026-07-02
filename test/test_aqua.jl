@testitem "Aqua.jl" begin
    using Aqua
    # Load the ColorVectorSpace extension so the ambiguity check covers its
    # `_lincomb_style`/`_lincomb2` methods against the core fallbacks (without
    # this the ext is absent here and the check is vacuous for it).
    using ColorTypes, ColorVectorSpace
    Aqua.test_all(
        FastInterpolations;
        # 55 false positives: Julia's has_unbound_tpars flags TypeVars nested
        # inside Vararg (NTuple{N, AbstractVector{Tg}}, NTuple{N, Tg}, etc.).
        # All are internal _-prefixed functions with no runtime impact.
        unbound_args = (broken = true,),
    )
end
