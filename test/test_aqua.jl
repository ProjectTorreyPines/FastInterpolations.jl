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
        # Disabled: FI has no `__init__`/`Timer`/`@async`, so loading it
        # cannot leave a persistent task — the check is vacuous for FI and
        # could only ever flag a dependency (all stable across platforms).
        # It probes exit by spawning a `Pkg.precompile` subprocess and
        # allowing only `tmax` s for it to exit; on the 7 GB / 3-core macOS
        # runner (mem ~99% under parallel workers) that subprocess hangs
        # regardless of `tmax`, giving a flaky single-platform false
        # positive at ~5 min/run for zero signal.
        persistent_tasks = false,
    )
end
