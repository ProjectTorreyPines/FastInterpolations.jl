@testitem "Constant One-Shot Series" setup = [AllocConstants] begin
    x = collect(range(0.0, 1.0, 101))
    y_sin = sin.(2π .* x)
    y_cos = cos.(2π .* x)
    y_exp = exp.(x)

    sitp = constant_interp(x, Series(y_sin, y_cos, y_exp))

    @testset "Scalar: Tuple → NTuple" begin
        xq = 0.37
        vals = constant_interp(x, Series(y_sin, y_cos, y_exp), xq)
        ref = sitp(xq)
        @test vals isa Vector{Float64}
        @test vals ≈ ref
    end

    @testset "Scalar: Matrix → Vector" begin
        Y = hcat(y_sin, y_cos, y_exp)
        vals = constant_interp(x, Series(Y), 0.37)
        ref = sitp(0.37)
        @test vals isa Vector{Float64}
        @test vals ≈ ref
    end

    @testset "In-place scalar" begin
        out = zeros(3)
        ret = constant_interp!(out, x, Series(y_sin, y_cos, y_exp), 0.37)
        ref = sitp(0.37)
        @test ret === out
        @test out ≈ ref
    end

    @testset "Vector query" begin
        xqs = [0.1, 0.37, 0.5, 0.9]
        outs = constant_interp(x, Series(y_sin, y_cos, y_exp), xqs)
        @test length(outs) == 3
        for j in eachindex(xqs)
            ref = sitp(xqs[j])
            for k in 1:3
                @test outs[k][j] ≈ ref[k]
            end
        end
    end

    @testset "Side options" begin
        for side_opt in [NearestSide(), LeftSide(), RightSide()]
            sitp_s = constant_interp(x, Series(y_sin, y_cos); side = side_opt)
            vals = constant_interp(x, Series(y_sin, y_cos), 0.37; side = side_opt)
            ref = sitp_s(0.37)
            @test collect(vals) ≈ ref
        end
    end

    @testset "Extrapolation modes" begin
        xq_oob = 1.5
        @test_throws DomainError constant_interp(x, Series(y_sin, y_cos), xq_oob)

        vals_clamp = constant_interp(x, Series(y_sin, y_cos), xq_oob; extrap = ClampExtrap())
        ref_sin = constant_interp(x, y_sin, xq_oob; extrap = ClampExtrap())
        ref_cos = constant_interp(x, y_cos, xq_oob; extrap = ClampExtrap())
        @test vals_clamp[1] ≈ ref_sin
        @test vals_clamp[2] ≈ ref_cos

        vals_ext = constant_interp(x, Series(y_sin, y_cos), xq_oob; extrap = ExtendExtrap())
        ref_sin_ext = constant_interp(x, y_sin, xq_oob; extrap = ExtendExtrap())
        @test vals_ext[1] ≈ ref_sin_ext

        # WrapExtrap
        vals_wrap = constant_interp(x, Series(y_sin, y_cos), xq_oob; extrap = WrapExtrap())
        ref_wrap_sin = constant_interp(x, y_sin, xq_oob; extrap = WrapExtrap())
        ref_wrap_cos = constant_interp(x, y_cos, xq_oob; extrap = WrapExtrap())
        @test vals_wrap[1] ≈ ref_wrap_sin
        @test vals_wrap[2] ≈ ref_wrap_cos

        # FillExtrap
        vals_fill = constant_interp(x, Series(y_sin, y_cos), xq_oob; extrap = FillExtrap(999.0))
        @test vals_fill[1] ≈ 999.0
        @test vals_fill[2] ≈ 999.0
    end

    @testset "Derivative ops" begin
        for d in 0:2
            op = DerivOp(d)
            vals = constant_interp(x, Series(y_sin, y_cos), 0.37; deriv = op)
            ref_sin = constant_interp(x, y_sin, 0.37; deriv = op)
            ref_cos = constant_interp(x, y_cos, 0.37; deriv = op)
            @test vals[1] ≈ ref_sin
            @test vals[2] ≈ ref_cos
        end
    end

    @testset "Float32" begin
        x32 = Float32.(x)
        y32_sin = Float32.(y_sin)
        y32_cos = Float32.(y_cos)
        vals = constant_interp(x32, Series(y32_sin, y32_cos), 0.5f0)
        @test vals isa Vector{Float32}
        ref_sin = constant_interp(x32, y32_sin, 0.5f0)
        @test vals[1] ≈ ref_sin
    end

    @testset "Complex values" begin
        y_c1 = complex.(y_sin, y_cos)
        y_c2 = complex.(y_exp, -y_exp)
        vals = constant_interp(x, Series(y_c1, y_c2), 0.5)
        ref1 = constant_interp(x, y_c1, 0.5)
        ref2 = constant_interp(x, y_c2, 0.5)
        @test vals[1] ≈ ref1
        @test vals[2] ≈ ref2
    end

    # ForwardDiff AD: skipped for constant interp (piecewise constant is
    # discontinuous, and _ConstantAnchoredQuery uses Tg(xi_primal) which
    # does not support Dual types)

    # Function-barrier pattern: pass outer vars as args (avoids @testset
    # try/catch type-instability under @testitem fresh-module compilation).
    @testset "Zero allocation (in-place scalar)" begin
        function measure(x, y_sin, y_cos)
            s = Series(y_sin, y_cos)
            out = zeros(2)
            constant_interp!(out, x, s, 0.5)  # warmup
            return @allocated constant_interp!(out, x, s, 0.5)
        end
        @test measure(x, y_sin, y_cos) <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation (in-place vector)" begin
        function measure(x, y_sin, y_cos)
            s = Series(y_sin, y_cos)
            xqs = [0.1, 0.37, 0.5, 0.9]
            outputs = [zeros(length(xqs)) for _ in 1:2]
            constant_interp!(outputs, x, s, xqs)  # warmup
            return @allocated constant_interp!(outputs, x, s, xqs)
        end
        @test measure(x, y_sin, y_cos) <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation (in-place vector, sorted len≥8 → LinearBinarySearch arm)" begin
        # A sorted query of length ≥ 8 flips `_is_likely_monotone` to true, so the
        # default AutoSearch resolves to the LinearBinarySearch/RefHint arm — the
        # adaptive branch the 4-query test above never reaches (len < 8 ⇒ BinarySearch).
        # The policy is chosen inside `_fill_series_anchors_resolved!`, so the Union
        # never reaches the build loop and the batch stays zero-alloc on this arm too.
        function measure(x, y_sin, y_cos)
            s = Series(y_sin, y_cos)
            xqs = collect(range(0.05, 0.95, 16))   # sorted, length 16 ≥ 8
            outputs = [zeros(length(xqs)) for _ in 1:2]
            constant_interp!(outputs, x, s, xqs)  # warmup
            constant_interp!(outputs, x, s, xqs)  # second warmup (JIT settle under @testitem)
            return @allocated constant_interp!(outputs, x, s, xqs)
        end
        @test measure(x, y_sin, y_cos) <= ALLOC_THRESHOLD
    end

    @testset "Eltype contract: fully-Int chain stays Int" begin
        # Series routes through the canonical kernel-shape trait, so the
        # output eltype promotes via `xq - xL`. The "stays Int" contract
        # requires a *fully-Int* chain (Int grid + Int y + Int xq). Float
        # xq with Int grid would (correctly) widen to Float — pinned in
        # test_constant_eltype.jl's "Float xq carrier" testset.
        x_i = 0:4
        y1_int = [0, 1, 3, 4, 7]
        y2_int = [2, 3, 1, 0, 5]
        vals = constant_interp(x_i, Series(y1_int, y2_int), 2)
        @test vals isa Vector{Int}
        @test vals[1] == constant_interp(x_i, y1_int, 2)
    end

    @testset "Eltype contract: FillExtrap fill_value must be compatible with y eltype" begin
        # Fully-Int chain so the output buffer is Vector{Int}; mismatched
        # Float fill_value then surfaces as InexactError on assign. Float
        # grid/xq would promote the buffer and silently absorb the Float
        # fill — that case is not what this contract tries to catch.
        x_i = 0:4
        y_int = [0, 1, 3, 4, 7]
        # Int series + Int fill_value → output stays Int, fill stores cleanly.
        vals = constant_interp(x_i, Series(y_int), 5; extrap = FillExtrap(9))
        @test vals isa Vector{Int}
        @test vals[1] == 9
        # Int series + Float fill_value → InexactError on Int-buffer assign.
        @test_throws InexactError constant_interp(x_i, Series(y_int), 5; extrap = FillExtrap(0.5))
    end

    @testset "Scalar: Vector-of-Vectors" begin
        vals = constant_interp(x, Series([y_sin, y_cos, y_exp]), 0.37)
        ref = constant_interp(x, Series(y_sin, y_cos, y_exp), 0.37)
        @test vals ≈ ref
    end

    @testset "DimensionMismatch on wrong output size" begin
        s = Series(y_sin, y_cos)
        out_wrong = zeros(5)
        @test_throws DimensionMismatch constant_interp!(out_wrong, x, s, 0.5)
    end

    @testset "PeriodicBC — scalar, :inclusive" begin
        s = Series(y_sin, y_cos)
        vals = constant_interp(x, s, 0.37; bc = PeriodicBC())
        vals_wrap = constant_interp(x, s, 0.37 + 1.0; bc = PeriodicBC())
        @test vals ≈ vals_wrap atol = 1.0e-12
    end

    @testset "PeriodicBC — scalar, :exclusive FVM" begin
        xc = [0.5, 1.5, 2.5]
        s = Series([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        vals_in = constant_interp(xc, s, 2.4; bc)
        @test vals_in[1] ≈ 30.0
        @test vals_in[2] ≈ 3.0
    end

    @testset "PeriodicBC — scalar in-place" begin
        s = Series(y_sin, y_cos)
        out = zeros(2)
        constant_interp!(out, x, s, 0.37; bc = PeriodicBC())
        @test out ≈ constant_interp(x, s, 0.37; bc = PeriodicBC())
    end

    @testset "PeriodicBC — vector in-place, :inclusive" begin
        s = Series(y_sin, y_cos)
        xqs = [0.1, 0.37, 0.9]
        outputs = [zeros(length(xqs)) for _ in 1:2]
        constant_interp!(outputs, x, s, xqs; bc = PeriodicBC())
        outputs_wrap = [zeros(length(xqs)) for _ in 1:2]
        constant_interp!(outputs_wrap, x, s, xqs .+ 1.0; bc = PeriodicBC())
        @test outputs[1] ≈ outputs_wrap[1] atol = 1.0e-12
        @test outputs[2] ≈ outputs_wrap[2] atol = 1.0e-12
    end

    @testset "PeriodicBC — vector in-place, :exclusive" begin
        xc = [0.5, 1.5, 2.5]
        s = Series([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        xqs = [0.6, 1.4, 2.4]
        outputs = [zeros(3), zeros(3)]
        constant_interp!(outputs, xc, s, xqs; bc)
        @test outputs[1] ≈ [10.0, 20.0, 30.0]   # NearestSide picks closest cell center
        @test outputs[2] ≈ [1.0, 2.0, 3.0]
    end

    @testset "PeriodicBC — vector allocating" begin
        s = Series(y_sin, y_cos)
        outs = constant_interp(x, s, [0.1, 0.37]; bc = PeriodicBC())
        @test length(outs) == 2 && length(outs[1]) == 2
    end

    @testset "PeriodicBC — :inclusive endpoint mismatch raises" begin
        s = Series(y_sin, y_exp)
        @test_throws ArgumentError constant_interp(x, s, 0.37; bc = PeriodicBC())
    end

    @testset "PeriodicBC — :exclusive period too small raises" begin
        xv = [0.0, 1.0, 2.0, 3.0]
        s = Series([0.0, 1.0, 2.0, 3.0])
        bc_bad = PeriodicBC(endpoint = :exclusive, period = 2.5)
        @test_throws ArgumentError constant_interp(xv, s, 1.5; bc = bc_bad)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # K×Q (large-NQ) loop-order path. `_series_use_kq_loop(NQ, K)` returns
    # true when `NQ > 16` (or `K ≥ 256`), routing to `_constant_series_batch_kq!`
    # — pool-acquired anchor vector + K-outer×Q-inner stream. The QK helper
    # (above tests at `Vector query`, `PeriodicBC — vector in-place ...`) only
    # exercises small NQ; these testsets lock in the KQ branches.
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Vector batch — K×Q loop (NQ=20 > threshold)" begin
        # NQ=20 triggers the kq! path for any K (since 20 > _SERIES_BATCH_NQ_THRESHOLD=16).
        NQ = 20
        xqs = collect(range(0.05, 0.95, NQ))
        s = Series(y_sin, y_cos, y_exp)

        @testset "Non-periodic NoBC — KQ result == QK reference" begin
            outputs_kq = [zeros(NQ) for _ in 1:3]
            constant_interp!(outputs_kq, x, s, xqs)
            # Cross-check each entry against the scalar oneshot path.
            for j in eachindex(xqs)
                ref = constant_interp(x, s, xqs[j])
                for k in 1:3
                    @test outputs_kq[k][j] ≈ ref[k] atol = 1.0e-12
                end
            end
        end

        @testset "Non-periodic with WrapExtrap on OOB queries" begin
            # Include a few OOB queries so `wrap = extrap_eff isa WrapExtrap`
            # branch fires inside `_constant_series_batch_kq!`.
            xqs_wrap = vcat(collect(range(0.05, 0.95, NQ - 4)), [-0.2, -0.05, 1.05, 1.2])
            outputs_kq = [zeros(length(xqs_wrap)) for _ in 1:2]
            constant_interp!(
                outputs_kq, x, Series(y_sin, y_cos), xqs_wrap;
                extrap = WrapExtrap()
            )
            for j in eachindex(xqs_wrap)
                ref = constant_interp(x, Series(y_sin, y_cos), xqs_wrap[j]; extrap = WrapExtrap())
                @test outputs_kq[1][j] ≈ ref[1] atol = 1.0e-12
                @test outputs_kq[2][j] ≈ ref[2] atol = 1.0e-12
            end
        end

        @testset "PeriodicBC{:inclusive} batch — KQ" begin
            outputs_kq = [zeros(NQ) for _ in 1:2]
            constant_interp!(outputs_kq, x, Series(y_sin, y_cos), xqs; bc = PeriodicBC())
            # Equivalent wrap-by-period query → bit-exact via shared anchor.
            outputs_kq_wrap = [zeros(NQ) for _ in 1:2]
            constant_interp!(
                outputs_kq_wrap, x, Series(y_sin, y_cos), xqs .+ 1.0; bc = PeriodicBC()
            )
            @test outputs_kq[1] ≈ outputs_kq_wrap[1] atol = 1.0e-12
            @test outputs_kq[2] ≈ outputs_kq_wrap[2] atol = 1.0e-12
        end

        @testset "PeriodicBC{:exclusive} FVM batch — KQ" begin
            xc = [0.5, 1.5, 2.5]
            s_fvm = Series([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
            bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
            xqs_fvm = collect(range(0.1, 2.9, NQ))   # NQ=20 across one period
            outputs_kq = [zeros(NQ), zeros(NQ)]
            constant_interp!(outputs_kq, xc, s_fvm, xqs_fvm; bc)
            # Cross-check each entry against the scalar in-place path.
            out_scalar = zeros(2)
            for j in eachindex(xqs_fvm)
                constant_interp!(out_scalar, xc, s_fvm, xqs_fvm[j]; bc)
                @test outputs_kq[1][j] ≈ out_scalar[1] atol = 1.0e-12
                @test outputs_kq[2][j] ≈ out_scalar[2] atol = 1.0e-12
            end
        end

        @testset "Range grid batch — KQ" begin
            # Range grid → `_to_float` produces `_CachedRange`. Forces
            # `_resolve_axis` + `_resolve_search` Range-specific branches
            # inside the kq! body.
            x_range = range(0.0, 1.0, 101)
            outputs_kq = [zeros(NQ) for _ in 1:2]
            constant_interp!(outputs_kq, x_range, Series(y_sin, y_cos), xqs)
            for j in eachindex(xqs)
                ref = constant_interp(x_range, Series(y_sin, y_cos), xqs[j])
                @test outputs_kq[1][j] ≈ ref[1] atol = 1.0e-12
                @test outputs_kq[2][j] ≈ ref[2] atol = 1.0e-12
            end
        end

        @testset "Float32 grid + Float64 query — KQ Tqp promotion" begin
            # Hits the `Tqp = promote_type(Tg_actual, Tq)` path inside kq!.
            x32 = Float32.(x)
            y32_sin = Float32.(y_sin)
            y32_cos = Float32.(y_cos)
            outputs_kq = [zeros(Float64, NQ) for _ in 1:2]
            constant_interp!(outputs_kq, x32, Series(y32_sin, y32_cos), xqs)
            for j in eachindex(xqs)
                ref = constant_interp(x32, Series(y32_sin, y32_cos), xqs[j])
                @test outputs_kq[1][j] ≈ ref[1] rtol = 1.0e-6
                @test outputs_kq[2][j] ≈ ref[2] rtol = 1.0e-6
            end
        end

        @testset "Derivative op + KQ" begin
            outputs_kq = [zeros(NQ) for _ in 1:2]
            constant_interp!(
                outputs_kq, x, Series(y_sin, y_cos), xqs;
                deriv = DerivOp(1)
            )
            # Constant interp derivative is zero everywhere except at sample
            # points (where it's a delta); for any non-sample interior query
            # the result must be zero.
            for j in eachindex(xqs)
                @test outputs_kq[1][j] == 0.0
                @test outputs_kq[2][j] == 0.0
            end
        end
    end

    @testset "Vector batch — K×Q via K threshold (K ≥ 256)" begin
        # Many small series — `K ≥ 256` triggers kq! regardless of NQ.
        # Use small NQ=4 so the entry is solely via the K-threshold branch.
        NQ = 4
        K = 256
        xqs = [0.1, 0.37, 0.5, 0.9]
        ys = [sin.(2π .* x .+ 0.01 * k) for k in 1:K]
        s = Series(ys)
        outputs_kq = [zeros(NQ) for _ in 1:K]
        constant_interp!(outputs_kq, x, s, xqs)
        # Spot-check a few entries against the scalar oneshot path.
        for j in eachindex(xqs)
            ref = constant_interp(x, s, xqs[j])
            for k in (1, K ÷ 2, K)
                @test outputs_kq[k][j] ≈ ref[k] atol = 1.0e-12
            end
        end
    end
end
