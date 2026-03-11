# ========================================
# ND Query Fallbacks (Query Protocol)
# ========================================
#
# Catch-all entry points for one-shot and adjoint APIs.
# Normalizes unrecognized query types via _resolve_queries,
# then delegates to the existing typed methods.
#
# These must be included AFTER all interpolant types are loaded,
# since they reference concrete one-shot and adjoint functions.

# ── One-shot ND fallbacks ──
# Each function forwards kwargs via kw..., so different per-type
# kwargs (bc, deriv, side, etc.) all pass through correctly.

for f in (:cubic_interp, :quadratic_interp, :linear_interp, :constant_interp)
    f! = Symbol(f, :!)
    @eval begin
        # Allocating one-shot: resolve queries → delegate to typed method
        function $f(
                grids::NTuple{N, AbstractVector},
                data::AbstractArray,
                queries;  # untyped fallback
                kw...
            ) where {N}
            canonical, _ = _resolve_queries(queries, Val(N))
            return $f(grids, data, canonical; kw...)
        end

        # In-place one-shot: resolve queries → delegate to typed method
        function $f!(
                output::AbstractVector,
                grids::NTuple{N, AbstractVector},
                data::AbstractArray,
                queries;  # untyped fallback
                kw...
            ) where {N}
            canonical, _ = _resolve_queries(queries, Val(N))
            return $f!(output, grids, data, canonical; kw...)
        end
    end
end

# ── Adjoint ND fallback ──
# cubic_adjoint AoS method (line 776) converts to SoA internally.
# We do the same: resolve → if AoS canonical, convert to SoA → delegate.

function cubic_adjoint(
        grids::NTuple{N, AbstractVector},
        queries;  # untyped fallback
        kw...
    ) where {N}
    canonical, mode = _resolve_queries(queries, Val(N))
    if mode isa AoSBatch
        # Convert AoS → SoA (same as existing AoS method in cubic_nd_adjoint.jl)
        soa_queries = ntuple(d -> map(q -> q[d], canonical), Val(N))
        return cubic_adjoint(grids, soa_queries; kw...)
    else
        return cubic_adjoint(grids, canonical; kw...)
    end
end
