# ========================================
# Duck-grid contract: source lint (ducktype-grid phase 6)
# ========================================
# Enforces the relaxed grid/query type contract: no `<:Real` bounds on grid
# (`Tg`), query (`Tq*`), or spacing (`Td`) params — and no `xq::Real`-style
# query args — in `src/`, OUTSIDE the enumerated allowlist below. Without this
# lint the next PR can silently reintroduce a bound (exactly how the original
# 285 `<:Real` sites accreted).

@testitem "Duck grid: source lint — Real bounds on grid/query params" begin
    src_dir = dirname(pathof(FastInterpolations))

    banned = [
        r"\bTg\s*<:\s*Real\b",
        r"\bTq[0-9A-Za-z_]*\s*<:\s*Real\b",
        r"\bTd\s*<:\s*Real\b",
        r"\bxq::Real\b",
        r"\bx0::Real\b",
        r"\bx1::Real\b",
        r"\bxi::Real\b",
        # Codex-review additions: spellings that slipped the name-based net.
        r"\bq::Real\b",
        r"\ba::Real\b",
        r"\bb::Real\b",
        r"\b(q|x_query|xq|queries|x_targets)::AbstractArray\{<:Real\}",
        r"\bquery::AbstractVector\{<:Real\}",
    ]

    # ── Enumerated allowlist (design doc § Durability) ──
    # Each entry: predicate on (relpath, line). One reason per class.
    allowed(rel, line) = (
        # 1. Adjoint / rrule paths keep <:Real (Dual machinery — separate project).
        occursin("adjoint", rel) ||
            # 2. Index-space demotion-gate arms: reachable only for T<:Real grids
            #    by construction (`_to_float` gates); value≡index space holds there.
            (rel == joinpath("core", "search.jl") && occursin("xq::Real", line)) ||
            # 2b. `_CachedRange` axis-search fast arms: generic siblings serve
            #     duck queries (ND unit-range eval/integrate verified GREEN);
            #     relaxing would reroute units onto index-leaning fast paths.
            (rel == joinpath("core", "nd_utils.jl") && occursin("q::Real", line)) ||
            # 2c. Extrap carrier Real arms: deliberate split — historic
            #     `zero(xq)*zero(val)` carrier keeps Int results Int; the duck
            #     arms add the dimensionless `inv(oneunit(xq))` factor.
            (rel == joinpath("core", "utils.jl") && occursin("_promote_extrap", line)) ||
            # 3. `_inv_const` Real arm — dispatch pair with the dimensionless arm.
            occursin("_inv_const", line) ||
            # 3b. Type-level units-branch idiom (`Tg <: Real || return _*_units(...)`).
            occursin("<: Real ||", line) ||
            # 4. DEFERRED: Series EVAL + anchored-query paths (integrate-side series
            #    sigs ARE relaxed; eval-side is a tracked follow-up, plan notes).
            occursin("series", rel) || occursin("anchor", rel) ||
            # 5. DEFERRED: public `coeffs`/derivative-view query args (follow-up).
            rel == "coeffs.jl" || rel == "derivative_view.jl"
    )

    violations = String[]
    allowed_hits = Dict{String, Int}()
    for (root, _, files) in walkdir(src_dir), f in files
        endswith(f, ".jl") || continue
        rel = relpath(joinpath(root, f), src_dir)
        in_doc = false
        for (i, line) in enumerate(eachline(joinpath(root, f)))
            # A one-line docstring (`"""…"""`) has an EVEN delimiter count → no net toggle;
            # only an odd count opens/closes a multi-line doc. (Old code toggled once per line
            # with any delimiter, so a one-liner flipped `in_doc` and skipped code to EOF.)
            if occursin("\"\"\"", line)
                isodd(count("\"\"\"", line)) && (in_doc = !in_doc)
                continue
            end
            in_doc && continue
            ls = lstrip(line)
            startswith(ls, "#") && continue
            for pat in banned
                occursin(pat, line) || continue
                if allowed(rel, line)
                    allowed_hits[rel] = get(allowed_hits, rel, 0) + 1
                else
                    push!(violations, "$(rel):$(i): $(rstrip(line))")
                end
            end
        end
    end

    if !isempty(violations)
        @info "duck-grid lint violations" violations
    end
    @test isempty(violations)

    # ── Ratchet: file-level allowlist classes must not absorb NEW debt ──
    # The broad predicates above (whole series/anchor files, coeffs.jl, …)
    # would otherwise mask fresh `<:Real` reintroductions inside those files.
    # Any change to these counts — up OR down — must be conscious: update the
    # table together with the source change (down = progress, shrink the entry;
    # up = new debt, justify it in the PR).
    expected_hits = Dict(
        "coeffs.jl" => 4,
        "constant/constant_adjoint.jl" => 1,
        "constant/constant_anchor.jl" => 3,
        "constant/constant_oneshot_series.jl" => 5,
        "constant/constant_series_interp.jl" => 3,
        "constant/nd/constant_nd_adjoint.jl" => 1,
        "core/anchor_common.jl" => 3,
        "core/nd_utils.jl" => 2,
        "core/search.jl" => 4,
        "core/series_lean_anchors.jl" => 1,
        "core/series_utils.jl" => 6,
        "core/utils.jl" => 7,
        "cubic/cubic_adjoint.jl" => 1,
        "cubic/cubic_anchor.jl" => 4,
        "cubic/cubic_oneshot.jl" => 3,
        "cubic/cubic_oneshot_series.jl" => 5,
        "cubic/cubic_series_interp.jl" => 4,
        "cubic/nd/cubic_nd_adjoint.jl" => 1,
        "derivative_view.jl" => 2,
        "hetero/hetero_adjoint.jl" => 1,
        "linear/linear_adjoint.jl" => 1,
        "linear/linear_anchor.jl" => 6,
        "linear/linear_oneshot_series.jl" => 5,
        "linear/linear_series_interp.jl" => 4,
        "linear/nd/linear_nd_adjoint.jl" => 1,
        "quadratic/nd/quadratic_nd_adjoint.jl" => 1,
        "quadratic/quadratic_anchor.jl" => 8,
        "quadratic/quadratic_interpolant.jl" => 1,
        "quadratic/quadratic_oneshot.jl" => 2,
        "quadratic/quadratic_oneshot_series.jl" => 4,
        "quadratic/quadratic_series_interp.jl" => 4,
    )
    if allowed_hits != expected_hits
        drift = [
            "$(k): expected $(get(expected_hits, k, 0)), got $(get(allowed_hits, k, 0))"
                for k in union(keys(expected_hits), keys(allowed_hits))
                if get(expected_hits, k, 0) != get(allowed_hits, k, 0)
        ]
        @info "duck-grid lint ratchet drift (update table consciously)" drift
    end
    @test allowed_hits == expected_hits
end
