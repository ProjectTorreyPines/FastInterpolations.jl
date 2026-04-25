# Forwarder smoke tests for the ND convenience entry points
# `pchip_interp` / `cardinal_interp` / `akima_interp` (and their `!` variants).
#
# These functions are 1D-native; the file under test
# (`src/hetero/local_hermite_nd_forward.jl`) only adds ND-shape methods that
# forward to `interp(grids, data; method = ...)`. So this test deliberately
# does NOT re-validate ND interpolation accuracy (covered by
# `test_hermite_onthefly.jl`, `test_hetero_nd.jl`, etc.). Instead it verifies:
#
#   1. Equivalence — forwarder result matches the direct `interp` result for
#      every one of the four ND call shapes (build, scalar oneshot, batch
#      oneshot, batch in-place). Scalar paths use `===` (ULP-exact identity);
#      batch/array paths use `==` since array identity is meaningless here.
#   2. Type stability — `@inferred` on the scalar oneshot path through each
#      forwarder. This is the path where any kwarg-passthrough Union would
#      surface most visibly.
#   3. Cardinal `tension` passthrough — non-default `tension` must reach
#      `CardinalInterp(tension)` (otherwise results would silently match the
#      default, defeating the kwarg).

@testitem "Local Hermite ND forwarders (pchip/cardinal/akima)" begin

    # ── shared 2D fixture ──
    x2 = collect(range(0.0, 2π, 15))
    y2 = collect(range(0.0, π, 10))
    data2 = [sin(xi) * cos(yj) for xi in x2, yj in y2]
    q2_scalar = (1.234, 0.789)
    q2_batch = [(0.3, 0.4), (1.5, 1.0), (2.0, 2.5), (5.5, 0.1)]

    # ── 3D fixture (verifies N>2 path of NTuple{N, ...}) ──
    x3 = collect(range(0.0, 1.0, 8))
    y3 = collect(range(0.0, 1.0, 7))
    z3 = collect(range(0.0, 1.0, 6))
    data3 = [sin(xi + yj) * cos(zk) for xi in x3, yj in y3, zk in z3]
    q3_scalar = (0.3, 0.6, 0.55)
    q3_batch = [(0.1, 0.2, 0.3), (0.5, 0.5, 0.5), (0.9, 0.8, 0.7)]

    # ── method list: (forwarder, in-place forwarder, method instance, label) ──
    methods_2d = (
        (pchip_interp, pchip_interp!, PchipInterp(), "pchip"),
        (cardinal_interp, cardinal_interp!, CardinalInterp(), "cardinal"),
        (akima_interp, akima_interp!, AkimaInterp(), "akima"),
    )

    @testset "Equivalence with direct `interp` — 2D ($label)" for (fwd, fwd!, m, label) in methods_2d
        # 1. Build path
        itp_fwd = fwd((x2, y2), data2)
        itp_dir = interp((x2, y2), data2; method = m)
        @test typeof(itp_fwd) === typeof(itp_dir)
        @test itp_fwd(q2_scalar) === itp_dir(q2_scalar)

        # 2. Scalar oneshot
        v_fwd = fwd((x2, y2), data2, q2_scalar)
        v_dir = interp((x2, y2), data2, q2_scalar; method = m)
        @test v_fwd === v_dir

        # 3. Batch oneshot
        b_fwd = fwd((x2, y2), data2, q2_batch)
        b_dir = interp((x2, y2), data2, q2_batch; method = m)
        @test b_fwd == b_dir

        # 4. Batch in-place
        out_fwd = similar(b_fwd)
        out_dir = similar(b_dir)
        fwd!(out_fwd, (x2, y2), data2, q2_batch)
        interp!(out_dir, (x2, y2), data2, q2_batch; method = m)
        @test out_fwd == out_dir
    end

    @testset "Equivalence with direct `interp` — 3D ($label)" for (fwd, fwd!, m, label) in methods_2d
        # Build + scalar oneshot are sufficient to prove the N=3 path works.
        itp_fwd = fwd((x3, y3, z3), data3)
        itp_dir = interp((x3, y3, z3), data3; method = m)
        @test typeof(itp_fwd) === typeof(itp_dir)
        @test itp_fwd(q3_scalar) === itp_dir(q3_scalar)

        v_fwd = fwd((x3, y3, z3), data3, q3_scalar)
        v_dir = interp((x3, y3, z3), data3, q3_scalar; method = m)
        @test v_fwd === v_dir

        b_fwd = fwd((x3, y3, z3), data3, q3_batch)
        b_dir = interp((x3, y3, z3), data3, q3_batch; method = m)
        @test b_fwd == b_dir
    end

    @testset "Type stability — scalar oneshot ($label)" for (fwd, _, _, label) in methods_2d
        # The scalar oneshot path is the canonical "fast" entry point —
        # if kwarg passthrough boxes anything, @inferred catches it here.
        @test (@inferred fwd((x2, y2), data2, q2_scalar)) isa Float64
        @test (@inferred fwd((x3, y3, z3), data3, q3_scalar)) isa Float64
    end

    @testset "Cardinal `tension` kwarg passthrough" begin
        # If the forwarder dropped `tension`, both calls would use the default
        # (0.0 = Catmull-Rom) and produce identical results. We need them to
        # differ to prove `tension` reaches `CardinalInterp(tension)`.
        v_default = cardinal_interp((x2, y2), data2, q2_scalar)
        v_tense = cardinal_interp((x2, y2), data2, q2_scalar; tension = 0.5)
        @test v_default !== v_tense

        # Same check on the build path.
        itp_default = cardinal_interp((x2, y2), data2)
        itp_tense = cardinal_interp((x2, y2), data2; tension = 0.5)
        @test itp_default(q2_scalar) !== itp_tense(q2_scalar)

        # And on the batch in-place path.
        out_default = zeros(length(q2_batch))
        out_tense = zeros(length(q2_batch))
        cardinal_interp!(out_default, (x2, y2), data2, q2_batch)
        cardinal_interp!(out_tense, (x2, y2), data2, q2_batch; tension = 0.5)
        @test out_default != out_tense

        # Cross-check: forwarder with `tension=0.5` must equal direct
        # `interp` with `CardinalInterp(0.5)` (proves the value is threaded
        # through, not just "any non-default").
        v_dir = interp((x2, y2), data2, q2_scalar; method = CardinalInterp(0.5))
        @test v_tense === v_dir
    end

    # ── Grid-flavor coverage ─────────────────────────────────────────────
    # The forwarder signature is `NTuple{N, AbstractVector}`, so it must
    # accept all three real-world combinations: (a) all-Vector (already
    # covered above), (b) all-range (uniform grids — common path that hits
    # `_CachedRange` fast search), (c) mixed range+Vector (e.g. uniform time
    # axis + non-uniform spatial axis). Verify each flavor:
    #   1. forwards to the same concrete type as direct `interp`,
    #   2. produces the same value, and
    #   3. stays type-stable on the scalar oneshot path.
    @testset "Grid flavors (all-range, mixed range+vector)" begin
        # Same domain/data as the 2D fixture above, but built with different
        # grid types so ND interpolation results are directly comparable
        # across flavors.
        xr = range(0.0, 2π, 15)              # StepRangeLen
        yr = range(0.0, π, 10)              # StepRangeLen
        xv = collect(xr)                     # Vector{Float64}
        yv = collect(yr)                     # Vector{Float64}
        data_rr = [sin(xi) * cos(yj) for xi in xr, yj in yr]

        grid_flavors = (
            ("all-range", (xr, yr)),
            ("range × vector", (xr, yv)),
            ("vector × range", (xv, yr)),
        )

        @testset "$flavor — $label" for (flavor, grids) in grid_flavors,
                (fwd, fwd!, m, label) in methods_2d

            # 1. Build path: forwarder must produce the *same concrete type*
            #    as direct `interp` — proves grid eltype/range info is not
            #    silently widened or converted by the forwarder.
            itp_fwd = fwd(grids, data_rr)
            itp_dir = interp(grids, data_rr; method = m)
            @test typeof(itp_fwd) === typeof(itp_dir)
            @test itp_fwd(q2_scalar) === itp_dir(q2_scalar)

            # 2. Scalar oneshot equivalence
            v_fwd = fwd(grids, data_rr, q2_scalar)
            v_dir = interp(grids, data_rr, q2_scalar; method = m)
            @test v_fwd === v_dir

            # 3. Batch oneshot + in-place
            b_fwd = fwd(grids, data_rr, q2_batch)
            b_dir = interp(grids, data_rr, q2_batch; method = m)
            @test b_fwd == b_dir

            out_fwd = similar(b_fwd)
            out_dir = similar(b_dir)
            fwd!(out_fwd, grids, data_rr, q2_batch)
            interp!(out_dir, grids, data_rr, q2_batch; method = m)
            @test out_fwd == out_dir

            # 4. Type stability on the scalar oneshot path. Mixed-grid tuples
            #    are still concrete (e.g. `Tuple{StepRangeLen, Vector{Float64}}`)
            #    so inference must succeed end-to-end.
            @test (@inferred fwd(grids, data_rr, q2_scalar)) isa Float64
        end

        # Cross-flavor sanity: all-range and all-vector grids describe the
        # *same* domain, so forwarder results must agree to ULP. (This is a
        # property of `interp`, not of the forwarder per se, but verifying it
        # here catches the case where a forwarder accidentally promotes
        # `range → Vector` and triggers a different code path.)
        for (fwd, _, _, _) in methods_2d
            v_range = fwd((xr, yr), data_rr, q2_scalar)
            v_vector = fwd((xv, yv), data_rr, q2_scalar)
            @test v_range ≈ v_vector rtol = 1.0e-12
        end
    end
end
