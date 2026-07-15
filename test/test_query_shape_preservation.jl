# ============================================================
# Query-Shape-Preserving Forward Evaluation
# ============================================================
#
# Design: docs/design/query_shape_preserving_evaluation.md
# Plan:   docs/design/PLAN_query_shape_preserving_batch_evaluation.md
#
# Contract: every batch forward-evaluation surface preserves the logical
# `size` of its query container. A matrix query yields a matrix result; a
# shaped SoA `(qx::Matrix, qy::Matrix)` yields a matrix; in-place calls
# require the output to match `_query_size(queries)` exactly (not just length).
#
# Reference for correctness = the existing (tested) vector AoS/SoA path,
# reshaped: `reshape(itp(vec_of_points), sz)`. This pins the new shaped
# result against the flat path bit-for-bit.
#
# Phase 1 covers the query-shape protocol + the persistent N-D quadrant.

# ------------------------------------------------------------
# Phase 1 · Protocol: _query_size / _query_length / _query_extract / _query_eltype
# ------------------------------------------------------------
@testitem "query shape: protocol size/length/extract/eltype" begin
    import FastInterpolations: _query_size, _query_length, _query_extract, _query_eltype

    # --- AoS matrix of N-D points ---
    pts12 = [(0.1 * k, 0.2 * k) for k in 1:12]
    q_aos_mat = reshape(pts12, 3, 4)
    q_aos_3d = reshape([(0.1 * k, 0.2 * k) for k in 1:12], 2, 2, 3)

    @test _query_size(q_aos_mat) == (3, 4)
    @test _query_size(q_aos_3d) == (2, 2, 3)
    @test _query_length(q_aos_mat) == 12
    @test _query_length(q_aos_mat) == prod(_query_size(q_aos_mat))     # length == prod(size)
    # column-major linear extraction lands each point at its N-D slot
    @test _query_extract(q_aos_mat, 5) == q_aos_mat[5]

    # --- 1-D real matrix (each element one scalar coordinate) ---
    q1 = reshape([0.1, 0.2, 0.3, 0.4], 2, 2)
    @test _query_size(q1) == (2, 2)
    @test _query_length(q1) == 4
    @test _query_eltype(q1) == Float64

    # --- shaped SoA: tuple of coordinate matrices ---
    qx = [0.1 0.2; 0.3 0.4]
    qy = [0.5 0.6; 0.7 0.8]
    @test _query_size((qx, qy)) == (2, 2)
    @test _query_length((qx, qy)) == 4
    @test _query_extract((qx, qy), 3) == (qx[3], qy[3])               # column-major pairwise
    @test _query_eltype((qx, qy)) == Float64

    # --- Vector queries unchanged (regression pin) ---
    v = [0.1, 0.2, 0.3]
    @test _query_size(v) == (3,)
    @test _query_size([(0.1, 0.2), (0.3, 0.4)]) == (2,)              # vector AoS stays flat
    @test _query_size(([1.0, 2.0], [3.0, 4.0])) == (2,)             # vector SoA stays flat
end

# ------------------------------------------------------------
# Phase 1 · Shaped SoA validation + ndims (mandatory correctness peers)
# ------------------------------------------------------------
@testitem "query shape: shaped SoA validation + ndims" begin
    import FastInterpolations: _query_validate, _query_check_ndims

    qx = [0.1 0.2; 0.3 0.4]
    qy = [0.5 0.6; 0.7 0.8]

    # matching-size SoA: validation is a no-op, ndims recognizes tuple length
    @test _query_validate((qx, qy)) === nothing
    @test _query_check_ndims((qx, qy), Val(2)) === nothing            # must NOT throw spurious ndims

    # mismatched SIZE (equal length would not catch this) throws before evaluation
    qy_bad = reshape([0.5, 0.6, 0.7, 0.8], 4, 1)                      # same length (4), different size
    @test size(qx) != size(qy_bad)
    @test length(qx) == length(qy_bad)
    @test_throws DimensionMismatch _query_validate((qx, qy_bad))

    # wrong axis count for the interpolation dimension still errors clearly
    @test_throws DimensionMismatch _query_check_ndims((qx, qy), Val(3))
end

# ------------------------------------------------------------
# Phase 1 · Persistent N-D shape preservation (allocating + in-place)
# ------------------------------------------------------------
@testitem "query shape: persistent N-D shape preservation" begin
    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]

    # logical query points, 12 of them, laid out 3x4
    qx12 = [0.3, 0.7, 1.1, 1.5, 1.9, 0.4, 0.8, 1.2, 1.6, 0.2, 0.6, 1.0]
    qy12 = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.15, 0.35, 0.55]
    pts_vec = [(qx12[k], qy12[k]) for k in eachindex(qx12)]

    q_aos_mat = reshape(pts_vec, 3, 4)
    qx_mat = reshape(qx12, 3, 4)
    qy_mat = reshape(qy12, 3, 4)

    configs = (
        ("LinearInterpolantND", linear_interp((x, y), data)),
        ("ConstantInterpolantND", constant_interp((x, y), data)),
        ("QuadraticInterpolantND", quadratic_interp((x, y), data)),
        ("CubicInterpolantND", cubic_interp((x, y), data)),
        ("HeteroInterpolantND", interp((x, y), data; method = (CubicInterp(), LinearInterp()))),
    )

    @testset "$name" for (name, itp) in configs
        # reference = existing vector path, reshaped
        ref_aos = reshape(itp(pts_vec), 3, 4)
        ref_soa = reshape(itp((qx12, qy12)), 3, 4)

        @testset "allocating AoS matrix -> matrix" begin
            r = itp(q_aos_mat)
            @test r isa Matrix
            @test size(r) == (3, 4)
            @test r == ref_aos
        end

        @testset "allocating shaped SoA -> matrix" begin
            r = itp((qx_mat, qy_mat))
            @test r isa Matrix
            @test size(r) == (3, 4)
            @test r == ref_soa
        end

        @testset "in-place AoS matrix" begin
            out = Matrix{Float64}(undef, 3, 4)
            res = itp(out, q_aos_mat)
            @test res === out
            @test out == ref_aos
        end

        @testset "in-place shaped SoA" begin
            out = Matrix{Float64}(undef, 3, 4)
            res = itp(out, (qx_mat, qy_mat))
            @test res === out
            @test out == ref_soa
        end

        @testset "exact-size rejection (right length, wrong shape)" begin
            # a length-12 vector output is NOT a valid sink for a 3x4 query
            @test_throws DimensionMismatch itp(Vector{Float64}(undef, 12), q_aos_mat)
            @test_throws DimensionMismatch itp(Vector{Float64}(undef, 12), (qx_mat, qy_mat))
        end
    end
end

# ------------------------------------------------------------
# Phase 1 · Dispatch guards (regression pins — must stay green throughout)
# ------------------------------------------------------------
@testitem "query shape: dispatch guards" begin
    import FastInterpolations: GriddedQuery

    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]
    itp = linear_interp((x, y), data)

    # N-D Vector{<:Real} of length N is ONE coordinate point, not a batch
    p = itp([0.5, 0.5])
    @test p isa Real
    @test p ≈ itp(([0.5], [0.5]))[1]

    # Vector{NTuple} is a vector batch -> Vector result
    batch = itp([(0.3, 0.1), (0.7, 0.3)])
    @test batch isa Vector
    @test length(batch) == 2

    # GriddedQuery keeps Cartesian-product shape (separable fast path)
    tx = [0.3, 0.7, 1.1]
    ty = [0.1, 0.5]
    g = itp(GriddedQuery((tx, ty)))
    @test size(g) == (3, 2)
    @test g[2, 1] ≈ itp([tx[2], ty[1]])

    # 1-D vector query still returns a Vector
    itp1 = linear_interp(x, sin.(x))
    r1 = itp1([0.3, 0.7, 1.1])
    @test r1 isa Vector
    @test length(r1) == 3
end

# ------------------------------------------------------------
# Phase 1 · Persistent N-D edge cases (deriv / extrap / view / precision)
# ------------------------------------------------------------
@testitem "query shape: persistent N-D edge cases" begin
    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]

    qx12 = [0.3, 0.7, 1.1, 1.5, 1.9, 0.4, 0.8, 1.2, 1.6, 0.2, 0.6, 1.0]
    qy12 = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.15, 0.35, 0.55]
    pts_vec = [(qx12[k], qy12[k]) for k in eachindex(qx12)]
    q_aos_mat = reshape(pts_vec, 3, 4)
    qx_mat = reshape(qx12, 3, 4)
    qy_mat = reshape(qy12, 3, 4)

    # Extrap MODE is a build-time contract (call-time `extrap=` only accepts an
    # InBounds fast-path override), so build a NoExtrap default and a Clamp variant.
    @testset "$name" for (name, ctor) in (("Linear", linear_interp), ("Cubic", cubic_interp))
        itp = ctor((x, y), data)                                  # default extrap = NoExtrap
        itp_clamp = ctor((x, y), data; extrap = ClampExtrap())

        @testset "derivative preserves shape" begin
            ref = reshape(itp((qx12, qy12); deriv = DerivOp(1, 0)), 3, 4)
            r = itp((qx_mat, qy_mat); deriv = DerivOp(1, 0))
            @test size(r) == (3, 4)
            @test r == ref
        end

        @testset "NoExtrap validates whole batch BEFORE any write" begin
            q_bad = copy(pts_vec)
            q_bad[1] = (5.0, 0.5)                     # single OOB point
            q_bad_mat = reshape(q_bad, 3, 4)
            out = fill(-999.0, 3, 4)                  # sentinel
            @test_throws DomainError itp(out, q_bad_mat)
            @test all(==(-999.0), out)                # untouched — pre-scan threw first
        end

        @testset "ClampExtrap value-match with OOB point" begin
            q_mix = copy(pts_vec)
            q_mix[1] = (5.0, 0.5)                      # OOB x — clamped, not thrown
            q_mix_mat = reshape(q_mix, 3, 4)
            @test itp_clamp(q_mix_mat) == reshape(itp_clamp(q_mix), 3, 4)
        end

        @testset "noncontiguous view query preserves shape" begin
            qv = @view q_aos_mat[1:2, :]              # 2×4 SubArray of points
            r = itp(qv)
            @test size(r) == (2, 4)
            @test r == reshape(itp(vec(collect(qv))), 2, 4)
        end

        @testset "mixed precision (Float32 SoA) preserves shape" begin
            qxf = Float32.(qx_mat)
            qyf = Float32.(qy_mat)
            r = itp((qxf, qyf))
            @test size(r) == (3, 4)
            # shaped-Float32 matches the Float32 VECTOR path exactly (same precision, reshaped)
            @test r == reshape(itp((vec(qxf), vec(qyf))), 3, 4)
        end
    end
end

# ------------------------------------------------------------
# Phase 1 · Allocation parity — shaped in-place is 0 B warm (matches vector lanes)
# ------------------------------------------------------------
@testitem "query shape: persistent N-D allocation parity" setup = [AllocConstants] begin
    # Function barrier: warmup then measure with every input as a TYPED argument, so a
    # boxed-local tuple construction at the call site is not counted as interpolation
    # allocation (the @allocated must see only the shaped in-place call itself).
    measure(itp, out, q) = (itp(out, q); itp(out, q); @allocated itp(out, q))

    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]
    itp = linear_interp((x, y), data)

    qx12 = [0.3, 0.7, 1.1, 1.5, 1.9, 0.4, 0.8, 1.2, 1.6, 0.2, 0.6, 1.0]
    qy12 = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.15, 0.35, 0.55]
    pts_vec = [(qx12[k], qy12[k]) for k in eachindex(qx12)]
    q_aos_mat = reshape(pts_vec, 3, 4)
    soa_mat = (reshape(qx12, 3, 4), reshape(qy12, 3, 4))
    soa_vec = (qx12, qy12)

    out_mat = Matrix{Float64}(undef, 3, 4)
    out_vec = Vector{Float64}(undef, 12)

    # shaped in-place: 0 B warm (matrix AoS + matrix SoA)
    @test measure(itp, out_mat, q_aos_mat) <= ND_ALLOC_THRESHOLD
    @test measure(itp, out_mat, soa_mat) <= ND_ALLOC_THRESHOLD
    # existing vector lanes unchanged: 0 B warm (regression pin — matches master)
    @test measure(itp, out_vec, pts_vec) <= ND_ALLOC_THRESHOLD
    @test measure(itp, out_vec, soa_vec) <= ND_ALLOC_THRESHOLD
end

# ============================================================
# Phase 2 · One-shot N-D quadrant (unified + dedicated)
# ============================================================

@testitem "query shape: one-shot N-D unified (interp/interp!)" begin
    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]
    m = (CubicInterp(), LinearInterp())          # heterogeneous method tuple

    qx12 = [0.3, 0.7, 1.1, 1.5, 1.9, 0.4, 0.8, 1.2, 1.6, 0.2, 0.6, 1.0]
    qy12 = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.15, 0.35, 0.55]
    pts_vec = [(qx12[k], qy12[k]) for k in eachindex(qx12)]
    q_aos_mat = reshape(pts_vec, 3, 4)
    soa_mat = (reshape(qx12, 3, 4), reshape(qy12, 3, 4))

    ref_aos = reshape(interp((x, y), data, pts_vec; method = m), 3, 4)
    ref_soa = reshape(interp((x, y), data, (qx12, qy12); method = m), 3, 4)

    @testset "allocating preserves shape" begin
        r1 = interp((x, y), data, q_aos_mat; method = m)
        @test r1 isa Matrix && size(r1) == (3, 4) && r1 == ref_aos
        r2 = interp((x, y), data, soa_mat; method = m)
        @test r2 isa Matrix && size(r2) == (3, 4) && r2 == ref_soa
    end

    @testset "in-place preserves shape" begin
        out = Matrix{Float64}(undef, 3, 4)
        @test interp!(out, (x, y), data, q_aos_mat; method = m) === out
        @test out == ref_aos
        out2 = Matrix{Float64}(undef, 3, 4)
        interp!(out2, (x, y), data, soa_mat; method = m)
        @test out2 == ref_soa
    end

    @testset "exact-size rejection (right length, wrong shape)" begin
        @test_throws DimensionMismatch interp!(Vector{Float64}(undef, 12), (x, y), data, q_aos_mat; method = m)
        @test_throws DimensionMismatch interp!(Vector{Float64}(undef, 12), (x, y), data, soa_mat; method = m)
    end
end

@testitem "query shape: one-shot N-D dedicated core families" begin
    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]

    qx12 = [0.3, 0.7, 1.1, 1.5, 1.9, 0.4, 0.8, 1.2, 1.6, 0.2, 0.6, 1.0]
    qy12 = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.15, 0.35, 0.55]
    pts_vec = [(qx12[k], qy12[k]) for k in eachindex(qx12)]
    q_aos_mat = reshape(pts_vec, 3, 4)
    soa_mat = (reshape(qx12, 3, 4), reshape(qy12, 3, 4))

    # (name, allocating_fn, inplace_fn) — cubic ND one-shot has no dedicated in-place
    families = (
        ("linear", linear_interp, linear_interp!),
        ("constant", constant_interp, constant_interp!),
        ("quadratic", quadratic_interp, quadratic_interp!),
        ("cubic", cubic_interp, cubic_interp!),
    )

    @testset "$name" for (name, f, f!) in families
        ref_aos = reshape(f((x, y), data, pts_vec), 3, 4)
        ref_soa = reshape(f((x, y), data, (qx12, qy12)), 3, 4)

        @testset "allocating preserves shape" begin
            r1 = f((x, y), data, q_aos_mat)
            @test r1 isa Matrix && size(r1) == (3, 4) && r1 == ref_aos
            r2 = f((x, y), data, soa_mat)
            @test r2 isa Matrix && size(r2) == (3, 4) && r2 == ref_soa
        end

        if f! !== nothing
            @testset "in-place preserves shape" begin
                out = Matrix{Float64}(undef, 3, 4)
                @test f!(out, (x, y), data, q_aos_mat) === out
                @test out == ref_aos
                out2 = Matrix{Float64}(undef, 3, 4)
                f!(out2, (x, y), data, soa_mat)
                @test out2 == ref_soa
            end

            @testset "exact-size rejection" begin
                @test_throws DimensionMismatch f!(Vector{Float64}(undef, 12), (x, y), data, q_aos_mat)
            end
        end
    end
end

@testitem "query shape: one-shot N-D local-Hermite + user-Hermite" begin
    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]

    qx12 = [0.3, 0.7, 1.1, 1.5, 1.9, 0.4, 0.8, 1.2, 1.6, 0.2, 0.6, 1.0]
    qy12 = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.15, 0.35, 0.55]
    pts_vec = [(qx12[k], qy12[k]) for k in eachindex(qx12)]
    q_aos_mat = reshape(pts_vec, 3, 4)
    soa_mat = (reshape(qx12, 3, 4), reshape(qy12, 3, 4))

    @testset "$name (local-Hermite forwarder)" for (name, f, f!) in (
            ("pchip", pchip_interp, pchip_interp!),
            ("cardinal", cardinal_interp, cardinal_interp!),
            ("akima", akima_interp, akima_interp!),
        )
        ref_aos = reshape(f((x, y), data, pts_vec), 3, 4)
        ref_soa = reshape(f((x, y), data, (qx12, qy12)), 3, 4)

        r1 = f((x, y), data, q_aos_mat)
        @test r1 isa Matrix && size(r1) == (3, 4) && r1 == ref_aos
        @test f((x, y), data, soa_mat) == ref_soa

        out = Matrix{Float64}(undef, 3, 4)
        @test f!(out, (x, y), data, q_aos_mat) === out
        @test out == ref_aos
        out2 = Matrix{Float64}(undef, 3, 4)
        f!(out2, (x, y), data, soa_mat)
        @test out2 == ref_soa

        @test_throws DimensionMismatch f!(Vector{Float64}(undef, 12), (x, y), data, q_aos_mat)
    end

    @testset "user-Hermite (explicit partials)" begin
        dfdx = [cos(xi) * cos(yj) for xi in x, yj in y]
        dfdy = [-sin(xi) * sin(yj) for xi in x, yj in y]
        d2 = [-cos(xi) * sin(yj) for xi in x, yj in y]
        p = HermitePartials((1, 0) => dfdx, (0, 1) => dfdy, (1, 1) => d2)

        ref_aos = reshape(hermite_interp((x, y), data, p, pts_vec), 3, 4)
        ref_soa = reshape(hermite_interp((x, y), data, p, (qx12, qy12)), 3, 4)

        r1 = hermite_interp((x, y), data, p, q_aos_mat)
        @test r1 isa Matrix && size(r1) == (3, 4) && r1 == ref_aos
        @test hermite_interp((x, y), data, p, soa_mat) == ref_soa

        out = Matrix{Float64}(undef, 3, 4)
        @test hermite_interp!(out, (x, y), data, p, q_aos_mat) === out
        @test out == ref_aos

        @test_throws DimensionMismatch hermite_interp!(Vector{Float64}(undef, 12), (x, y), data, p, q_aos_mat)
    end
end

@testitem "query shape: one-shot N-D allocation parity" setup = [AllocConstants] begin
    # Function barriers (query passed as a typed arg — see Phase-1 alloc parity note).
    meas_u(out, grids, data, q, m) = (interp!(out, grids, data, q; method = m); interp!(out, grids, data, q; method = m); @allocated interp!(out, grids, data, q; method = m))
    meas_d(out, grids, data, q) = (linear_interp!(out, grids, data, q); linear_interp!(out, grids, data, q); @allocated linear_interp!(out, grids, data, q))

    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]
    m = (CubicInterp(), LinearInterp())

    qx12 = [0.3, 0.7, 1.1, 1.5, 1.9, 0.4, 0.8, 1.2, 1.6, 0.2, 0.6, 1.0]
    qy12 = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.15, 0.35, 0.55]
    pts_vec = [(qx12[k], qy12[k]) for k in eachindex(qx12)]
    q_aos_mat = reshape(pts_vec, 3, 4)
    soa_mat = (reshape(qx12, 3, 4), reshape(qy12, 3, 4))

    out_mat = Matrix{Float64}(undef, 3, 4)
    out_vec = Vector{Float64}(undef, 12)

    # unified shaped in-place: 0 B warm (matrix AoS + matrix SoA); vector unchanged
    @test meas_u(out_mat, (x, y), data, q_aos_mat, m) <= ND_ALLOC_THRESHOLD
    @test meas_u(out_mat, (x, y), data, soa_mat, m) <= ND_ALLOC_THRESHOLD
    @test meas_u(out_vec, (x, y), data, pts_vec, m) <= ND_ALLOC_THRESHOLD
    # dedicated shaped in-place: 0 B warm; vector unchanged
    @test meas_d(out_mat, (x, y), data, q_aos_mat) <= ND_ALLOC_THRESHOLD
    @test meas_d(out_mat, (x, y), data, soa_mat) <= ND_ALLOC_THRESHOLD
    @test meas_d(out_vec, (x, y), data, pts_vec) <= ND_ALLOC_THRESHOLD
end

# ============================================================
# Phase 3 · Persistent 1-D quadrant + DerivativeView
# ============================================================
# Reference = `map(itp, q)` (scalar eval per point, naturally shape-preserving).

@testitem "query shape: persistent 1-D shape preservation" begin
    x = collect(range(0.0, 2π, 25))
    y = sin.(x)
    dy = cos.(x)

    q12 = collect(range(0.3, 6.0, 12))
    q_mat = reshape(q12, 3, 4)
    q_3d = reshape(collect(range(0.3, 6.0, 24)), 2, 3, 4)

    families = (
        ("linear", linear_interp(x, y)),
        ("constant", constant_interp(x, y)),
        ("quadratic", quadratic_interp(x, y)),
        ("cubic", cubic_interp(x, y)),
        ("pchip", pchip_interp(x, y)),
        ("akima", akima_interp(x, y)),
        ("cardinal", cardinal_interp(x, y)),
        ("hermite", hermite_interp(x, y, dy)),
    )

    @testset "$name" for (name, itp) in families
        @testset "allocating Matrix / 3-D" begin
            r = itp(q_mat)
            @test r isa Matrix && size(r) == (3, 4)
            @test r == map(itp, q_mat)
            r3 = itp(q_3d)
            @test size(r3) == (2, 3, 4)
            @test r3 == map(itp, q_3d)
        end

        @testset "in-place Matrix" begin
            out = Matrix{Float64}(undef, 3, 4)
            @test itp(out, q_mat) === out
            @test out == map(itp, q_mat)
        end

        @testset "noncontiguous view query" begin
            qv = @view q_mat[1:2, :]
            r = itp(qv)
            @test size(r) == (2, 4)
            @test r == map(itp, qv)
        end

        @testset "exact-size rejection" begin
            @test_throws DimensionMismatch itp(Vector{Float64}(undef, 12), q_mat)
        end

        @testset "N=1 SoA (q,) parity" begin
            @test itp((q_mat,)) == itp(q_mat)
            out = Matrix{Float64}(undef, 3, 4)
            itp(out, (q_mat,))
            @test out == itp(q_mat)
        end
    end
end

@testitem "query shape: 1-D DerivativeView shaped" begin
    x = collect(range(0.0, 2π, 25))
    y = sin.(x)
    itp = cubic_interp(x, y)

    q12 = collect(range(0.3, 6.0, 12))
    q_mat = reshape(q12, 3, 4)

    @testset "deriv$k preserves shape (allocating + in-place)" for k in 1:3
        dv = (deriv1, deriv2, deriv3)[k](itp)
        r = dv(q_mat)
        @test r isa Matrix && size(r) == (3, 4)
        @test r == map(dv, q_mat)

        out = Matrix{Float64}(undef, 3, 4)
        @test dv(out, q_mat) === out
        @test out == map(dv, q_mat)

        @test_throws DimensionMismatch dv(Vector{Float64}(undef, 12), q_mat)
    end
end

@testitem "query shape: persistent 1-D allocation parity" setup = [AllocConstants] begin
    measure(itp, out, q) = (itp(out, q); itp(out, q); @allocated itp(out, q))

    x = collect(range(0.0, 2π, 25))
    y = sin.(x)
    itp = cubic_interp(x, y)

    q12 = collect(range(0.3, 6.0, 12))
    q_mat = reshape(q12, 3, 4)
    qv = @view q_mat[1:2, :]

    out_mat = Matrix{Float64}(undef, 3, 4)
    out_view = Matrix{Float64}(undef, 2, 4)
    out_vec = Vector{Float64}(undef, 12)

    # shaped in-place: 0 B warm (dense Matrix + noncontiguous view)
    @test measure(itp, out_mat, q_mat) <= ALLOC_THRESHOLD
    @test measure(itp, out_view, qv) <= ALLOC_THRESHOLD
    # existing vector lane unchanged: 0 B warm
    @test measure(itp, out_vec, q12) <= ALLOC_THRESHOLD
end
