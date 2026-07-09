@testitem "Aqua.jl" begin
    using Aqua
    # Load the ColorVectorSpace extension so the ambiguity check covers its
    # `_linear_blend_style`/`_linear_value_blend` methods against the core
    # fallbacks (without this the ext is absent here and the check is
    # vacuous for it).
    using ColorTypes, ColorVectorSpace
    Aqua.test_all(
        FastInterpolations;
        # false positives (count drifts with internal methods): Julia's
        # has_unbound_tpars flags TypeVars nested inside Vararg
        # (NTuple{N, AbstractVector{Tg}}, NTuple{N, Tg}, etc.). All are
        # internal _-prefixed functions with no runtime impact.
        unbound_args = (broken = true,),
        # The check spawns a subprocess that builds+precompiles a dummy
        # package depending on FI; Aqua's 10 s default is too tight on a
        # loaded CI runner (parallel workers, 7 GB macOS). A REAL leaked
        # task hangs the subprocess forever, so it still fails at any tmax —
        # raising it only removes the timeout false positive.
        persistent_tasks = (tmax = 300,),
    )
end
