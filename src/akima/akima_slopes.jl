# ========================================
# Akima Slope Computation (Akima 1970)
# ========================================
# Weighted average of adjacent secants.
# Outlier-robust: gives less weight to deviant secants.
# O(n), single pass, in-place.
#
# Reference: Akima, "A New Method of Interpolation and Smooth Curve Fitting
# Based on Local Procedures", JACM 17(4), 1970, pp. 589-602.
#
# Requires n ≥ 3 (at least 2 secants). For n=2, degenerates to linear.

"""
    _akima_slopes!(dy, x, y)

Compute Akima slopes in-place.

# Arguments
- `dy::AbstractVector{Tv}`: output slopes (pre-allocated, length n)
- `x::AbstractVector{Tg}`: grid points (sorted, length n, ≥ 2)
- `y::AbstractVector{Tv}`: function values (length n)

# Algorithm
The Akima formula uses 4 adjacent secants to compute each interior slope:

    m[k] = (y[k+1] - y[k]) / (x[k+1] - x[k])      (secant slopes)

    w1 = |m[k+1] - m[k]|
    w2 = |m[k-1] - m[k-2]|

    if w1 + w2 == 0
        d[k] = (m[k-1] + m[k]) / 2                   (equal weights fallback)
    else
        d[k] = (w1 * m[k-1] + w2 * m[k]) / (w1 + w2) (weighted average)
    end

Endpoints use virtual secants computed by linear extrapolation of the
secant sequence.

# Complexity
O(n), single pass, zero allocation (writes into `dy`).
"""
function _akima_slopes!(
        dy::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector;
        bc::AbstractBC = NoBC()
    ) where {Tg}
    n = length(x)
    @assert n >= 2 "Akima requires at least 2 points"
    @assert length(y) == n "y length must match x"
    @assert length(dy) == n "dy length must match x"

    # Special case: 2 points. PeriodicBC routes through the wrap-aware
    # 4-secant helper (cycle=2 for `:exclusive` yields a 2-secant alternation).
    if n == 2
        if bc isa PeriodicBC
            @inbounds dy[1] = _akima_local_4secant_periodic(x, y, 1, n, bc)
            @inbounds dy[2] = _akima_local_4secant_periodic(x, y, 2, n, bc)
            return dy
        end
        @inbounds begin
            δ = (y[2] - y[1]) / (x[2] - x[1])
            dy[1] = δ
            dy[2] = δ
        end
        return dy
    end

    # Special case: 3 points → simple average at interior, one-sided at endpoints
    if n == 3
        if bc isa PeriodicBC
            # Wrap-aware path: every index is within K=5 stencil reach of the
            # join, so use the closed-cycle 4-secant formula at all 3 points.
            @inbounds for i in 1:3
                dy[i] = _akima_local_4secant_periodic(x, y, i, n, bc)
            end
            return dy
        end
        @inbounds begin
            m1 = (y[2] - y[1]) / (x[2] - x[1])
            m2 = (y[3] - y[2]) / (x[3] - x[2])
            dy[1] = m1
            dy[2] = (m1 + m2) / 2
            dy[3] = m2
        end
        return dy
    end

    # General case: n ≥ 4
    # We need secants m[1:n-1] plus virtual secants m[-1], m[0], m[n], m[n+1]
    # for the 5-point stencil at boundaries. Compute on-the-fly with rolling window.

    # Compute secant slopes (store temporarily in dy to avoid allocation)
    # We'll use a rolling window approach instead.

    # First, compute the n-1 secant slopes into a conceptual array.
    # For endpoints, we need virtual secants computed by linear extrapolation
    # of the secant sequence: m[-1] = 2*m[1] - m[2], m[0] = 2*m[1] - m[2], etc.
    # Actually, Akima's original uses: m[-1] = 2*m[0] - m[1] where m[0] = 2*m[1] - m[2]
    # Simplification: extrapolate m sequence linearly.

    # Compute all n-1 secant slopes
    @inbounds m1 = (y[2] - y[1]) / (x[2] - x[1])
    @inbounds m2 = (y[3] - y[2]) / (x[3] - x[2])
    @inbounds m3 = (y[4] - y[3]) / (x[4] - x[3])

    # Boundary virtual / wrapped secants for the LEFT side of the domain.
    # NoBC:                          linear extrapolation (Akima's original).
    # PeriodicBC{:inclusive}:        m[-1]=m[n-2], m[0]=m[n-1] (closed cycle on n-1 cells).
    # PeriodicBC{:exclusive}:        m[-1]=m[n-1], m[0]=m[n]=seam = (y[1]-y[n])/seam_h
    #                                (closed cycle on n cells, with virtual seam cell).
    # The `_periodic_secant` abstraction absorbs both PeriodicBC variants.
    if bc isa PeriodicBC
        @inbounds m_0    = _periodic_secant(x, y, 0, n, bc)     # m[0]
        @inbounds m_neg1 = _periodic_secant(x, y, -1, n, bc)    # m[-1]
    else
        @inbounds m_0    = 2 * m1 - m2                          # virtual (NoBC)
        @inbounds m_neg1 = 3 * m1 - 2 * m2                      # virtual (NoBC)
    end

    # Left endpoint: k=1 uses m[-1], m[0], m[1], m[2].
    @inbounds dy[1] = _akima_weighted_slope(m_neg1, m_0, m1, m2)

    # k=2: uses m[0], m[1], m[2], m[3]. m[0] above is wrap-aware so this stencil
    # is fully closed-cycle accurate under PeriodicBC.
    @inbounds dy[2] = _akima_weighted_slope(m_0, m1, m2, m3)

    # Interior points: k=3 to n-2 — K=5 stencil entirely within real-secants
    # range [1, n-1]; no wrap. Rolling window unchanged.
    m_km2 = m1
    m_km1 = m2
    m_k = m3

    @inbounds for k in 3:(n - 2)
        m_kp1 = (y[k + 2] - y[k + 1]) / (x[k + 2] - x[k + 1])
        dy[k] = _akima_weighted_slope(m_km2, m_km1, m_k, m_kp1)
        m_km2 = m_km1
        m_km1 = m_k
        m_k = m_kp1
    end

    # After loop: m_km2 = m[n-3], m_km1 = m[n-2], m_k = m[n-1]

    # Boundary virtual / wrapped secants for the RIGHT side of the domain.
    # NoBC:                          linear extrapolation.
    # PeriodicBC{:inclusive}:        m[n]=m[1], m[n+1]=m[2] (closed cycle on n-1 cells).
    # PeriodicBC{:exclusive}:        m[n]=seam, m[n+1]=m[1] (closed cycle on n cells).
    if bc isa PeriodicBC
        @inbounds m_np1 = _periodic_secant(x, y, n, n, bc)         # m[n]
        @inbounds m_np2 = _periodic_secant(x, y, n + 1, n, bc)     # m[n+1]
    else
        m_np1 = 2 * m_k - m_km1                                    # virtual (NoBC)
        m_np2 = 3 * m_k - 2 * m_km1                                # virtual (NoBC)
    end

    # k=n-1: uses m[n-3], m[n-2], m[n-1], m[n]. m[n] is wrap-aware under PeriodicBC.
    @inbounds dy[n - 1] = _akima_weighted_slope(m_km2, m_km1, m_k, m_np1)

    # Right endpoint: k=n uses m[n-2], m[n-1], m[n], m[n+1].
    # For PeriodicBC{:inclusive}: this stencil equals the i=1 stencil under wrap
    # (m[n-2]=m[i-2 wrapped], etc.), so the result equals dy[1] automatically.
    # For PeriodicBC{:exclusive}: stencil is genuinely different (dy[1] ≠ dy[n]).
    @inbounds dy[n] = _akima_weighted_slope(m_km1, m_k, m_np1, m_np2)

    return dy
end

"""
    _akima_weighted_slope(m_km2, m_km1, m_k, m_kp1) -> slope

Compute Akima weighted-average slope from 4 adjacent secants.

    w1 = |m[k+1] - m[k]|
    w2 = |m[k-1] - m[k-2]|
    d  = (w1 * m[k-1] + w2 * m[k]) / (w1 + w2)

Falls back to simple average when w1 + w2 ≈ 0 (equal or nearly-equal secants).
"""
@inline function _akima_weighted_slope(m_km2::Tv, m_km1::Tv, m_k::Tv, m_kp1::Tv) where {Tv}
    w1 = abs(m_kp1 - m_k)
    w2 = abs(m_km1 - m_km2)
    wsum = w1 + w2
    if wsum == zero(wsum)
        return (m_km1 + m_k) / 2
    else
        return (w1 * m_km1 + w2 * m_k) / wsum
    end
end
