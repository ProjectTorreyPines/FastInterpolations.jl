# ========================================
# `_StatefulPayload` surface contract: source lint
# ========================================
# The stateful wrapper exists to keep the OOB state branch at eval time. Its
# derivative-scale type param `S` is an implementation detail of the anchor
# BUILD: eval arms recover the scale from `typeof(a.payload)`, so they must not
# name `S` in their signatures. Spelling `_StatefulPayload{P, S}` in a
# dispatch-only arm leaks a shared-wrapper param into every family's eval code —
# that is exactly what made a `{P}` → `{P, S}` change churn four families plus
# their test replicas. Dispatch arms use `<:_StatefulPayload{P}`, which matches
# any `S`; only the arms that CONSTRUCT the wrapper may name it.

@testitem "Stateful payload: only construction sites may name the scale param" begin
    src_dir = dirname(pathof(FastInterpolations))

    naming_S = Regex("_StatefulPayload\\{\\s*P\\s*,\\s*S\\s*\\}")
    calls_wrapper = Regex("_StatefulPayload\\{\\s*P\\s*,\\s*S\\s*\\}\\(")
    # A construction site either CALLS the wrapper (`…{P, S}(inner, state)`) or
    # RETURNS the wrapper type itself (the `_maybe_stateful_payload` type factory,
    # whose whole body is the bare type).
    returns_type(line) = strip(line) == "_StatefulPayload{P, S}"

    offenders = String[]
    construct_sites = String[]
    for (root, _, files) in walkdir(src_dir), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        rel = replace(relpath(path, src_dir), '\\' => '/')
        lines = readlines(path)
        for (i, line) in enumerate(lines)
            occursin(naming_S, line) || continue
            startswith(lstrip(line), "#") && continue
            # The call form may sit a few lines below the signature that names `S`.
            window = join(lines[i:min(i + 12, end)], "\n")
            if returns_type(line) || occursin(calls_wrapper, window)
                push!(construct_sites, "$(rel):$(i)")
            else
                push!(offenders, "$(rel):$(i): $(rstrip(line))")
            end
        end
    end

    if !isempty(offenders)
        @info "dispatch-only arms naming the scale param (use `<:_StatefulPayload{P}`)" offenders
    end
    @test isempty(offenders)

    # The construction sites are enumerated: `S` genuinely belongs there (the
    # anchor type is being built), and a change to this set must be conscious.
    @test sort(construct_sites) == [
        "constant/constant_series_payloads.jl:116",   # ConstantInterp stateful resolve
        "constant/constant_series_payloads.jl:122",
        "constant/constant_series_payloads.jl:123",
        "core/series_lean_anchors.jl:21",             # `_maybe_stateful_payload` type factory
        "core/series_lean_anchors.jl:33",             # generic stateful resolve
        "core/series_lean_anchors.jl:39",
        "core/series_lean_anchors.jl:40",
    ]
end
