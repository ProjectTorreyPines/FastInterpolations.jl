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
        r"\bxi::Real\b",
    ]

    # ── Enumerated allowlist (design doc § Durability) ──
    # Each entry: predicate on (relpath, line). One reason per class.
    allowed(rel, line) = (
        # 1. Adjoint / rrule paths keep <:Real (Dual machinery — separate project).
        occursin("adjoint", rel) ||
            # 2. Index-space demotion-gate arms: reachable only for T<:Real grids
            #    by construction (`_to_float` gates); value≡index space holds there.
            (rel == joinpath("core", "search.jl") && occursin("xq::Real", line)) ||
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
    for (root, _, files) in walkdir(src_dir), f in files
        endswith(f, ".jl") || continue
        rel = relpath(joinpath(root, f), src_dir)
        in_doc = false
        for (i, line) in enumerate(eachline(joinpath(root, f)))
            occursin("\"\"\"", line) && (in_doc = !in_doc; continue)
            in_doc && continue
            ls = lstrip(line)
            startswith(ls, "#") && continue
            for pat in banned
                occursin(pat, line) || continue
                allowed(rel, line) && continue
                push!(violations, "$(rel):$(i): $(rstrip(line))")
            end
        end
    end

    if !isempty(violations)
        @info "duck-grid lint violations" violations
    end
    @test isempty(violations)
end
