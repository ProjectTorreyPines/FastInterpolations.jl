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
            (rel == "core/search.jl" && occursin("xq::Real", line)) ||
            # 2b. `_CachedRange` axis-search fast arms: generic siblings serve
            #     duck queries (ND unit-range eval/integrate verified GREEN);
            #     relaxing would reroute units onto index-leaning fast paths.
            (rel == "core/nd_utils.jl" && occursin("q::Real", line)) ||
            # 2c. Extrap carrier Real arms: deliberate split — historic
            #     `zero(xq)*zero(val)` carrier keeps Int results Int; the duck
            #     arms add the dimensionless `inv(oneunit(xq))` factor.
            (rel == "core/utils.jl" && occursin("_promote_extrap", line)) ||
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
        # Normalize to forward slashes: `relpath` yields `\` on Windows, but the
        # `allowed` predicate and `expected_hits` ratchet key on `/` (Unix-native),
        # so without this the Dict keys never match on Windows and the test fails.
        rel = replace(relpath(joinpath(root, f), src_dir), '\\' => '/')
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
        # constant series + one-shot: relaxed to Number (unit-grid Series eval,
        # Codex follow-up) — were 3 + 5, now 0 → keys dropped.
        "constant/nd/constant_nd_adjoint.jl" => 1,
        # anchor_common.jl (`_anchor_loc` ×2 + `_AnchorLoc` struct): relaxed to
        # Number for the deferred anchored-query paths (ND GriddedQuery on unit
        # grids) — was 3, now 0 → key dropped.
        "core/nd_utils.jl" => 2,
        "core/search.jl" => 4,
        # series_lean_anchors.jl (`_build_series_anchor`): relaxed to Number for
        # unit-grid Series eval — was 1, now 0 → key dropped.
        # series_utils.jl (the 6 `_constant_extrap_boundary_value` /
        # `_fill_constant_extrap_simd!` OOB arms): the `Tq <: Real` bound was a
        # `<:Real`-era leftover that made EVERY Series batch OOB query throw a
        # MethodError on a unit grid (constant/quadratic/cubic — linear escaped
        # only because its carrier is the dimensionless α). Bodies already went
        # through `one(Tq)`, so relaxing needed no other change — was 6, now 0.
        "core/utils.jl" => 6,
        "cubic/cubic_adjoint.jl" => 1,
        "cubic/cubic_anchor.jl" => 4,
        "cubic/cubic_oneshot.jl" => 3,
        # cubic series + one-shot: eval sigs relaxed to Number for unit-grid cubic
        # Series (build nondimensionalizes like the scalar path) — one-shot was 5,
        # now 0 → key dropped; interp was 4, now 1 = the `Tg <: Real ||` units-branch
        # idiom itself (allowlist class 3b).
        "cubic/cubic_series_interp.jl" => 1,
        "cubic/nd/cubic_nd_adjoint.jl" => 1,
        # derivative_view.jl: in-place view query bounds relaxed to Number (Codex
        # #4) — was 2, now 0 → key dropped.
        "hetero/hetero_adjoint.jl" => 1,
        "linear/linear_adjoint.jl" => 1,
        "linear/linear_anchor.jl" => 6,
        # linear series + one-shot (eval sigs): relaxed to Number for unit-grid
        # Series eval (Codex #1) — now 0 → keys dropped.
        "linear/nd/linear_nd_adjoint.jl" => 1,
        "quadratic/nd/quadratic_nd_adjoint.jl" => 1,
        "quadratic/quadratic_anchor.jl" => 8,
        "quadratic/quadratic_oneshot.jl" => 2,
        # quadratic series + one-shot: eval sigs relaxed to Number (Codex parity;
        # eval reaches a documented solver-storage limit, but the SIG isn't the
        # blocker) — were 4 + 4, now 0 → keys dropped.
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

# ========================================
# Strip-twin ratchet (refac/duck-thomas)
# ========================================
# The strip→solve→reattach twin style (`_*_units` sibling builders,
# `_strip_*_units` helpers, `<: Real` reroute forks in the solver families) is
# banned; the counts below ratchet DOWN to zero as the duck-thomas phases
# delete each site, and any INCREASE is a regression. Exact-name matching only:
# `_check_nd_hessian_units`, `_strip_periodic_bc`, `_strip_wrap_extrap` are
# legitimate names a broad pattern would false-positive on.
#
# Phase schedule (update counts consciously at each phase commit):
#   P2 cubic scalar twin → P3 periodic → P4 quadratic scalar+Series twins
#   → P5 cubic Series twin → P6 one-shot reroute forks → all zeros.

@testitem "Duck grid: strip-twin ratchet — mention counts must only decrease" begin
    src_dir = dirname(pathof(FastInterpolations))

    expected_tokens = Dict(
        "_cubic_interp_units" => 0,      # P2: cubic scalar twin deleted
        "_cubic_series_units" => 2,
        "_quadratic_interp_units" => 0,  # P4: quadratic scalar twin deleted
        "_strip_series_bc_units" => 3,
        "_strip_bc_units" => 9,          # P4: defs beside their last consumer (cubic Series twin)
    )
    # `Tg <: Real || return …` / `if !(Tg <: Real)` REROUTE forks, counted only
    # inside the solver family trees (cubic/, quadratic/) — `utils.jl`'s
    # promotion arm and adjoint gating live elsewhere and are legitimate.
    # Reject-guards (`<: Real || throw(...)`) are exempt: throwing an
    # actionable error for a not-yet-native combo is the endorsed idiom, not a
    # parallel solve path (e.g. the periodic-units guard until Phase 3).
    fork_res = (r"<: Real \|\|", r"if !\([A-Za-z_][A-Za-z0-9_]* <: Real\)")
    expected_forks = 12   # P2: cubic surface fork gone; P4: quadratic surface + Series forks gone

    counts, forks = let counts = Dict(k => 0 for k in keys(expected_tokens)), forks = 0
        for (root, _, files) in walkdir(src_dir), f in files
            endswith(f, ".jl") || continue
            path = joinpath(root, f)
            rel = relpath(path, src_dir)
            in_family = startswith(rel, "cubic") || startswith(rel, "quadratic")
            for line in eachline(path)
                for k in keys(expected_tokens)
                    occursin(k, line) && (counts[k] += 1)
                end
                if in_family && (occursin(fork_res[1], line) || occursin(fork_res[2], line))
                    occursin("throw", line) || (forks += 1)   # reject-guard exemption
                end
            end
        end
        counts, forks
    end

    if counts != expected_tokens || forks != expected_forks
        @info "strip-twin ratchet drift (decreases: update table at the phase commit; increases: regression)" counts expected_tokens forks expected_forks
    end
    @test counts == expected_tokens
    @test forks == expected_forks
end
