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
# Testitems are grouped by shared compilation surface (each @testitem is a
# module that compiles fresh, so one-@testitem-per-case is slow); related
# families live under logical @testsets inside a handful of testitems.

# ------------------------------------------------------------
# Query-shape protocol + shaped-SoA validation (no interpolants — fast)
# ------------------------------------------------------------
@testitem "query shape: protocol + SoA validation" begin
    import FastInterpolations: _query_size, _query_length, _query_extract, _query_eltype,
        _query_validate, _query_check_ndims

    @testset "protocol size/length/extract/eltype" begin
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

    @testset "shaped SoA validation + ndims" begin
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
end

# ------------------------------------------------------------
# Persistent N-D: shape preservation + dispatch guards + edge cases
# ------------------------------------------------------------
@testitem "query shape: persistent N-D" begin
    import FastInterpolations: GriddedQuery

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

    @testset "shape preservation — $name" for (name, itp) in (
            ("LinearInterpolantND", linear_interp((x, y), data)),
            ("ConstantInterpolantND", constant_interp((x, y), data)),
            ("QuadraticInterpolantND", quadratic_interp((x, y), data)),
            ("CubicInterpolantND", cubic_interp((x, y), data)),
            ("HeteroInterpolantND", interp((x, y), data; method = (CubicInterp(), LinearInterp()))),
        )
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
            @test itp(out, q_aos_mat) === out
            @test out == ref_aos
        end
        @testset "in-place shaped SoA" begin
            out = Matrix{Float64}(undef, 3, 4)
            @test itp(out, (qx_mat, qy_mat)) === out
            @test out == ref_soa
        end
        @testset "exact-size rejection (right length, wrong shape)" begin
            # a length-12 vector output is NOT a valid sink for a 3x4 query
            @test_throws DimensionMismatch itp(Vector{Float64}(undef, 12), q_aos_mat)
            @test_throws DimensionMismatch itp(Vector{Float64}(undef, 12), (qx_mat, qy_mat))
        end
    end

    @testset "dispatch guards" begin
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

    # Extrap MODE is a build-time contract (call-time `extrap=` only accepts an
    # InBounds fast-path override), so build a NoExtrap default and a Clamp variant.
    @testset "edge cases — $name" for (name, ctor) in (("Linear", linear_interp), ("Cubic", cubic_interp))
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
# One-shot N-D: unified + dedicated cores + local/user-Hermite
# ------------------------------------------------------------
@testitem "query shape: one-shot N-D" begin
    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]

    qx12 = [0.3, 0.7, 1.1, 1.5, 1.9, 0.4, 0.8, 1.2, 1.6, 0.2, 0.6, 1.0]
    qy12 = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.15, 0.35, 0.55]
    pts_vec = [(qx12[k], qy12[k]) for k in eachindex(qx12)]
    q_aos_mat = reshape(pts_vec, 3, 4)
    soa_mat = (reshape(qx12, 3, 4), reshape(qy12, 3, 4))

    @testset "unified (interp/interp!)" begin
        m = (CubicInterp(), LinearInterp())          # heterogeneous method tuple
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

    @testset "dedicated core family — $name" for (name, f, f!) in (
            ("linear", linear_interp, linear_interp!),
            ("constant", constant_interp, constant_interp!),
            ("quadratic", quadratic_interp, quadratic_interp!),
            ("cubic", cubic_interp, cubic_interp!),
        )
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

    @testset "local-Hermite forwarder — $name" for (name, f, f!) in (
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

# ------------------------------------------------------------
# Persistent 1-D: 8 families, allocating + in-place shape preservation
# ------------------------------------------------------------
# Reference = `map(itp, q)` (scalar eval per point, naturally shape-preserving).
# Compared with `isclose` (ULP-tolerant): the batch and scalar kernels differ only
# by FMA/muladd contraction, which is inline- and Julia/LLVM-version dependent, so
# strict `==` flakes on some platforms (e.g. LTS). Shape/size is still pinned exactly.
@testitem "query shape: persistent 1-D" setup = [Basic] begin
    x = collect(range(0.0, 2π, 25))
    y = sin.(x)
    dy = cos.(x)

    q12 = collect(range(0.3, 6.0, 12))
    q_mat = reshape(q12, 3, 4)
    q_3d = reshape(collect(range(0.3, 6.0, 24)), 2, 3, 4)

    @testset "$name" for (name, itp) in (
            ("linear", linear_interp(x, y)),
            ("constant", constant_interp(x, y)),
            ("quadratic", quadratic_interp(x, y)),
            ("cubic", cubic_interp(x, y)),
            ("pchip", pchip_interp(x, y)),
            ("akima", akima_interp(x, y)),
            ("cardinal", cardinal_interp(x, y)),
            ("hermite", hermite_interp(x, y, dy)),
        )
        @testset "allocating Matrix / 3-D" begin
            r = itp(q_mat)
            @test r isa Matrix && size(r) == (3, 4)
            @test isclose(r, map(itp, q_mat))
            r3 = itp(q_3d)
            @test size(r3) == (2, 3, 4)
            @test isclose(r3, map(itp, q_3d))
        end
        @testset "in-place Matrix" begin
            out = Matrix{Float64}(undef, 3, 4)
            @test itp(out, q_mat) === out
            @test isclose(out, map(itp, q_mat))
        end
        @testset "noncontiguous view query" begin
            qv = @view q_mat[1:2, :]
            r = itp(qv)
            @test size(r) == (2, 4)
            @test isclose(r, map(itp, qv))
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

# ------------------------------------------------------------
# DerivativeView shaped in-place: 1-D (deriv1/2/3) + N-D mixed partial
# ------------------------------------------------------------
@testitem "query shape: DerivativeView shaped (1-D + N-D)" setup = [Basic] begin
    @testset "1-D deriv$k preserves shape (allocating + in-place)" for k in 1:3
        x = collect(range(0.0, 2π, 25))
        y = sin.(x)
        itp = cubic_interp(x, y)
        q_mat = reshape(collect(range(0.3, 6.0, 12)), 3, 4)

        dv = (deriv1, deriv2, deriv3)[k](itp)
        r = dv(q_mat)
        @test r isa Matrix && size(r) == (3, 4)
        # derivative kernels reassociate more than value kernels → wider ULP band
        @test isclose(r, map(dv, q_mat); nulps = 4096)

        out = Matrix{Float64}(undef, 3, 4)
        @test dv(out, q_mat) === out
        @test isclose(out, map(dv, q_mat); nulps = 4096)

        @test_throws DimensionMismatch dv(Vector{Float64}(undef, 12), q_mat)
    end

    @testset "N-D in-place shaped deriv" begin
        x = collect(range(0.0, 2.0, 11))
        y = collect(range(0.0, 1.0, 9))
        data = [sin(a) * cos(b) for a in x, b in y]
        dv = deriv_view(cubic_interp((x, y), data), (1, 0))   # ∂/∂x on a 2-D interpolant

        qx = [0.3, 0.7, 1.1, 1.5, 1.9, 0.4]
        qy = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2]
        pts = [(qx[k], qy[k]) for k in eachindex(qx)]
        aos_mat = reshape(pts, 2, 3)
        soa_mat = (reshape(qx, 2, 3), reshape(qy, 2, 3))

        # The feature: in-place shaped ND deriv eval. The allocating and in-place
        # paths forward to the same parent batch core, so results are bitwise
        # identical (shape preservation, not a fresh computation) → `==`.
        @testset "shaped in-place == allocating" for (q, shp) in
            ((aos_mat, (2, 3)), (soa_mat, (2, 3)), (pts, (6,)), ((qx, qy), (6,)))
            out = Array{Float64}(undef, shp)
            @test dv(out, q) === out
            @test size(dv(q)) == shp
            @test out == dv(q)
        end

        # exact-size rejection: a length-6 vector sink for a 2x3 query must throw.
        @test_throws DimensionMismatch dv(Vector{Float64}(undef, 6), aos_mat)

        # The three ND-specialized in-place methods behind these calls exist SOLELY to
        # satisfy `Aqua.test_ambiguities` — each tie-breaks the generic ND method against
        # an all-ITP `(::AbstractArray,::Real)` / `(::AbstractVector,::AbstractVector{<:Real})`
        # / `(::AbstractArray,::AbstractArray{<:Real})` form. A bare Real scalar/array is never
        # a valid N-D point query, so they reject with ArgumentError rather than forward an
        # ill-typed query; invoke directly for coverage (repo precedent:
        # test_nd_utils_shared.jl "N=0 Aqua disambiguators").
        @testset "Aqua tie-breaker guards reject non-point queries" begin
            @test_throws ArgumentError dv(Matrix{Float64}(undef, 2, 2), 0.5)
            @test_throws ArgumentError dv(Vector{Float64}(undef, 4), [0.1, 0.2, 0.3, 0.4])
            @test_throws ArgumentError dv(Matrix{Float64}(undef, 2, 2), [0.1 0.2; 0.3 0.4])
        end
    end
end

# ------------------------------------------------------------
# One-shot 1-D: core families + local-Hermite + unified
# ------------------------------------------------------------
# Reference = the vector one-shot path, reshaped.
@testitem "query shape: one-shot 1-D" begin
    x = collect(range(0.0, 2π, 25))
    y = sin.(x)
    dy = cos.(x)
    q12 = collect(range(0.3, 6.0, 12))
    q_mat = reshape(q12, 3, 4)

    @testset "core family — $name" for (name, f, f!) in (
            ("linear", q -> linear_interp(x, y, q), (o, q) -> linear_interp!(o, x, y, q)),
            ("constant", q -> constant_interp(x, y, q), (o, q) -> constant_interp!(o, x, y, q)),
            ("quadratic", q -> quadratic_interp(x, y, q), (o, q) -> quadratic_interp!(o, x, y, q)),
            ("cubic", q -> cubic_interp(x, y, q), (o, q) -> cubic_interp!(o, x, y, q)),
            ("hermite", q -> hermite_interp(x, y, dy, q), (o, q) -> hermite_interp!(o, x, y, dy, q)),
        )
        ref = reshape(f(q12), 3, 4)

        @testset "allocating preserves shape" begin
            r = f(q_mat)
            @test r isa Matrix && size(r) == (3, 4)
            @test r == ref
        end
        @testset "in-place preserves shape" begin
            o = Matrix{Float64}(undef, 3, 4)
            @test f!(o, q_mat) === o
            @test o == ref
        end
        @testset "exact-size rejection" begin
            @test_throws DimensionMismatch f!(Vector{Float64}(undef, 12), q_mat)
        end
    end

    @testset "local-Hermite — $name" for (name, f, f!) in (
            ("pchip", (q) -> pchip_interp(x, y, q), (o, q) -> pchip_interp!(o, x, y, q)),
            ("cardinal", (q) -> cardinal_interp(x, y, q), (o, q) -> cardinal_interp!(o, x, y, q)),
            ("akima", (q) -> akima_interp(x, y, q), (o, q) -> akima_interp!(o, x, y, q)),
        )
        ref = reshape(f(q12), 3, 4)
        r = f(q_mat)
        @test r isa Matrix && size(r) == (3, 4) && r == ref
        o = Matrix{Float64}(undef, 3, 4)
        @test f!(o, q_mat) === o
        @test o == ref
        @test_throws DimensionMismatch f!(Vector{Float64}(undef, 12), q_mat)
    end

    @testset "unified interp/interp!" begin
        m = CubicInterp()
        ref = reshape(interp(x, y, q12; method = m), 3, 4)
        r = interp(x, y, q_mat; method = m)
        @test r isa Matrix && size(r) == (3, 4) && r == ref
        o = Matrix{Float64}(undef, 3, 4)
        @test interp!(o, x, y, q_mat; method = m) === o
        @test o == ref
        @test_throws DimensionMismatch interp!(Vector{Float64}(undef, 12), x, y, q_mat; method = m)
    end
end

# ------------------------------------------------------------
# N=1 tuple-grid collapse + empty / degenerate shaped queries
# ------------------------------------------------------------
@testitem "query shape: N=1 collapse + empty/degenerate" begin
    x = collect(range(0.0, 2π, 25))
    y = sin.(x)
    q12 = collect(range(0.3, 6.0, 12))
    q_mat = reshape(q12, 3, 4)

    @testset "N=1 tuple-grid collapse — $name" for (name, fn, fn!) in (
            ("linear", linear_interp, linear_interp!),
            ("constant", constant_interp, constant_interp!),
            ("quadratic", quadratic_interp, quadratic_interp!),
            ("cubic", cubic_interp, cubic_interp!),
            ("pchip", pchip_interp, pchip_interp!),
        )
        native = fn(x, y, q_mat)                      # bare-grid shaped 1-D
        @test native isa Matrix && size(native) == (3, 4)
        @test fn((x,), y, q_mat) == native            # tuple-grid collapse preserves shape
        @test fn((x,), y, (q_mat,)) == native         # single-axis SoA collapse
        o = Matrix{Float64}(undef, 3, 4)
        fn!(o, (x,), y, q_mat)
        @test o == native
    end

    @testset "empty + degenerate shaped queries" begin
        yb = collect(range(0.0, 1.0, 11))

        # --- 1-D: empty matrix query → empty-shaped result ---
        itp1 = cubic_interp(x, y)
        qe = Matrix{Float64}(undef, 0, 3)
        @test size(itp1(qe)) == (0, 3)
        oe = Matrix{Float64}(undef, 0, 3)
        @test itp1(oe, qe) === oe
        @test itp1(Float64[]) == Float64[]                     # empty vector still a vector
        @test_throws DimensionMismatch itp1(Matrix{Float64}(undef, 1, 3), qe)   # wrong-shape sink

        # --- N-D: empty AoS matrix + empty SoA ---
        data = [sin(a) * cos(b) for a in x, b in yb]
        itp2 = linear_interp((x, yb), data)
        q_aos_e = Matrix{NTuple{2, Float64}}(undef, 0, 2)
        @test size(itp2(q_aos_e)) == (0, 2)
        soa_e = (Matrix{Float64}(undef, 0, 2), Matrix{Float64}(undef, 0, 2))
        @test size(itp2(soa_e)) == (0, 2)
        oe2 = Matrix{Float64}(undef, 0, 2)
        @test itp2(oe2, q_aos_e) === oe2
    end
end

# ------------------------------------------------------------
# Allocation parity — shaped in-place is 0 B warm (matches vector lanes)
# ------------------------------------------------------------
@testitem "query shape: allocation parity (0 B warm)" setup = [AllocConstants] begin
    # Function barriers kept at module scope (no captured/boxed locals): warmup
    # then measure with every input as a TYPED argument, so a boxed-local tuple
    # construction at the call site is not counted as interpolation allocation
    # (the @allocated must see only the shaped in-place call itself).
    measure(itp, out, q) = (itp(out, q); itp(out, q); @allocated itp(out, q))
    meas_u(out, grids, data, q, m) = (interp!(out, grids, data, q; method = m); interp!(out, grids, data, q; method = m); @allocated interp!(out, grids, data, q; method = m))
    meas_d(out, grids, data, q) = (linear_interp!(out, grids, data, q); linear_interp!(out, grids, data, q); @allocated linear_interp!(out, grids, data, q))
    mL(o, x, y, q) = (linear_interp!(o, x, y, q); linear_interp!(o, x, y, q); @allocated linear_interp!(o, x, y, q))
    mC(o, x, y, q) = (cubic_interp!(o, x, y, q); cubic_interp!(o, x, y, q); @allocated cubic_interp!(o, x, y, q))

    @testset "persistent N-D" begin
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

    @testset "one-shot N-D" begin
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

    @testset "persistent 1-D" begin
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

    @testset "one-shot 1-D" begin
        x = collect(range(0.0, 2π, 25))
        y = sin.(x)
        q12 = collect(range(0.3, 6.0, 12))
        q_mat = reshape(q12, 3, 4)
        om = Matrix{Float64}(undef, 3, 4)
        ov = Vector{Float64}(undef, 12)

        # shaped 1-D one-shot in-place: 0 B warm (matrix); vector lane unchanged
        @test mL(om, x, y, q_mat) <= ALLOC_THRESHOLD
        @test mL(ov, x, y, q12) <= ALLOC_THRESHOLD
        @test mC(om, x, y, q_mat) <= ALLOC_THRESHOLD
        @test mC(ov, x, y, q12) <= ALLOC_THRESHOLD
    end
end
