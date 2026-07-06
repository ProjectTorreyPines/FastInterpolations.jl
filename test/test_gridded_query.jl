# GriddedQuery separable 2D-linear evaluation + op-aware axis-anchor primitive.
# Design spec: claudedocs/design/2026-07-04-gridded-axisgeom-2d-linear-spec.md
# (main checkout; claudedocs is gitignored).

@testitem "_LinearAnchor primitive: forwarder, kernel pair, layout" begin
    using FastInterpolations
    using FastInterpolations: _LinearAnchor, _eval_anchor, _axis_anchor, _resolve_alpha,
        _anchor_loc, _linear_value_blend, LinearInterp, EvalValue, _AbstractAxisAnchor
    using InteractiveUtils: code_llvm

    # ── construction + named-field access through the Val forwarder ──────────
    a = _LinearAnchor{Float64, EvalValue}(3, (alpha = 0.25,))
    @test a isa _AbstractAxisAnchor
    @test a.idx == 3
    @test a.alpha == 0.25
    @test propertynames(a) == (:idx, :alpha)
    @test isbits(a)
    @test sizeof(a) == 16   # Int + Float64, NamedTuple layout == raw field layout

    # Float32 payload stays Float32 (no silent widening)
    a32 = _LinearAnchor{Float32, EvalValue}(2, (alpha = 0.5f0,))
    @test a32.alpha === 0.5f0
    @test sizeof(a32) == 12 || sizeof(a32) == 16   # padding is platform-defined

    # ── matched-pair kernel ≡ the underlying blend, bit-exact ────────────────
    yL, yR = 1.5, 4.5
    @test _eval_anchor(a, yL, yR) === _linear_value_blend(0.25, yL, yR)

    # ── scalar builder: locate → alpha → extrap fold ─────────────────────────
    g = collect(1.0:10.0)
    loc = _anchor_loc(g, 3.25, false)
    b = _axis_anchor(LinearInterp(), EvalValue(), loc, g, FastInterpolations.ExtendExtrap(), Float64)
    @test b.idx == 3
    @test b.alpha ≈ 0.25

    # Clamp folds OOB weight to the boundary node (left: idx=1, alpha=0)
    locL = _anchor_loc(g, 0.0, false)
    bL = _axis_anchor(LinearInterp(), EvalValue(), locL, g, FastInterpolations.ClampExtrap(), Float64)
    @test bL.idx == 1
    @test bL.alpha === 0.0
    # Extend keeps the out-of-range weight (linear extrapolation)
    bE = _axis_anchor(LinearInterp(), EvalValue(), locL, g, FastInterpolations.ExtendExtrap(), Float64)
    @test bE.alpha === -1.0

    # ── forwarder pin: Val-dispatch getproperty folds to plain getfield ──────
    # Twin struct with native fields = the reference codegen.
    # (Defined at testitem top level so code_llvm sees a concrete method.)
    kernP(x::_LinearAnchor{Float64, EvalValue}, l, r) = _eval_anchor(x, l, r)
    struct _TwinAnchor
        idx::Int
        alpha::Float64
    end
    kernC(x::_TwinAnchor, l, r) = _linear_value_blend(x.alpha, l, r)
    llP = sprint(io -> code_llvm(io, kernP, Tuple{typeof(a), Float64, Float64}; debuginfo = :none))
    llC = sprint(io -> code_llvm(io, kernC, Tuple{_TwinAnchor, Float64, Float64}; debuginfo = :none))
    @test count("br ", llP) == count("br ", llC)          # no forwarder branch survives
    @test count("select", llP) == count("select", llC)
    @test !occursin("jl_box", llP)                        # no boxing
    @test abs(count('\n', llP) - count('\n', llC)) <= 2   # same instruction count (± label noise)
end

@testitem "_AxisAnchorBatch: batch builder + identity flag" begin
    using FastInterpolations
    using FastInterpolations: _axis_anchors, _AxisAnchorBatch, _LinearAnchor,
        LinearInterp, EvalValue

    g = collect(range(0.0, 1.0, 11))     # nodes 0.0, 0.1, ..., 1.0

    # general targets
    t = [0.05, 0.55, 0.999]
    b = _axis_anchors(LinearInterp(), EvalValue(), g, t, ExtendExtrap(), 1)
    @test b isa _AxisAnchorBatch
    @test length(b) == 3
    @test eltype(b.anchors) <: _LinearAnchor{Float64, EvalValue}
    @test b.anchors[1].idx == 1 && b.anchors[1].alpha ≈ 0.5
    @test b.identity == false

    # identity: targets exactly the grid nodes → every (idx == k, alpha == 0)
    bid = _axis_anchors(LinearInterp(), EvalValue(), g, copy(g), ClampExtrap(), 1)
    @test bid.identity == true
    # same nodes but different length → NOT identity (elision needs M == n)
    bpart = _axis_anchors(LinearInterp(), EvalValue(), g, g[1:5], ClampExtrap(), 1)
    @test bpart.identity == false

    # Float32 grid+targets stay Float32
    g32 = collect(Float32.(g)); t32 = Float32[0.05, 0.55]
    b32 = _axis_anchors(LinearInterp(), EvalValue(), g32, t32, ClampExtrap(), 1)
    @test eltype(b32.anchors) <: _LinearAnchor{Float32, EvalValue}
end

@testitem "GriddedQuery correctness matrix vs point-wise" begin
    using FastInterpolations
    import FastInterpolations as FI

    # {F64, F32} × {Clamp, Extend} × {up, down, mixed, non-monotonic, M == 1}
    # (NoExtrap joins the matrix in the NoExtrap task.)
    function check(itp, tx, ty; rtol = 1.0e-14, atol = 1.0e-14)
        C = itp(GriddedQuery((tx, ty)))
        ref = [itp((x, y)) for x in tx, y in ty]
        @test eltype(C) == typeof(itp((first(tx), first(ty))))   # output-eltype pin
        @test size(C) == (length(tx), length(ty))
        @test all(isapprox.(C, ref; rtol, atol))
    end

    for TF in (Float64, Float32), ex in (ClampExtrap(), ExtendExtrap())
        rt = TF == Float64 ? 1.0e-14 : 2.0f-6
        A = rand(TF, 32, 24)
        itp = FI.linear_interp((collect(TF, 1:32), collect(TF, 1:24)), A; extrap = ex)
        up = (range(TF(1), TF(32), 57), range(TF(1), TF(24), 41))
        down = (range(TF(1), TF(32), 13), range(TF(1), TF(24), 9))
        mixed = (range(TF(1), TF(32), 57), range(TF(1), TF(24), 9))
        oob = (range(TF(0), TF(33), 20), range(TF(0.5), TF(24.5), 20))  # extrap exercise
        nonmono = (TF[7.3, 2.1, 30.9, 2.1], TF[11.0, 3.5])              # unsorted + repeat
        single = (TF[15.5], TF[8.25])                                    # M == 1
        for (tx, ty) in (up, down, mixed, oob, nonmono, single)
            check(itp, collect(tx), collect(ty); rtol = rt, atol = rt)
        end
    end

    # mixed per-axis extraps (Clamp × Extend)
    A = rand(16, 16)
    itp2 = FI.linear_interp((1.0:16.0, 1.0:16.0), A; extrap = (ClampExtrap(), ExtendExtrap()))
    tx = collect(range(0.0, 17.0, 21)); ty = collect(range(0.0, 17.0, 21))
    C = itp2(GriddedQuery((tx, ty)))
    ref = [itp2((x, y)) for x in tx, y in ty]
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))
end

@testitem "pass kernels + cost-model order" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: _axis_anchors, _pass_blend_dim2!, _pass_gather_dim1!,
        _gridded_dim2_first, LinearInterp, EvalValue

    A = rand(24, 20)
    itp = FI.linear_interp((1.0:24.0, 1.0:20.0), A; extrap = FI.ClampExtrap())
    tx = collect(range(1.0, 24.0, 37)); ty = collect(range(1.0, 20.0, 31))
    M, N = length(tx), length(ty)
    p1 = _axis_anchors(LinearInterp(), EvalValue(), itp.grids[1], tx, itp.extraps[1], 1)
    p2 = _axis_anchors(LinearInterp(), EvalValue(), itp.grids[2], ty, itp.extraps[2], 2)
    ref = [itp((x, y)) for x in tx, y in ty]

    # order A: blend dim2 (24×31 mid) then gather dim1
    B1 = Matrix{Float64}(undef, 24, N)
    CA = Matrix{Float64}(undef, M, N)
    _pass_gather_dim1!(CA, _pass_blend_dim2!(B1, A, p2), p1)
    # order B: gather dim1 (37×20 mid) then blend dim2
    B2 = Matrix{Float64}(undef, M, 20)
    CB = Matrix{Float64}(undef, M, N)
    _pass_blend_dim2!(CB, _pass_gather_dim1!(B2, A, p1), p2)

    # both orders are mathematically equivalent — machine-eps, NOT bit-identity
    @test all(isapprox.(CA, ref; rtol = 1.0e-14, atol = 1.0e-14))
    @test all(isapprox.(CB, ref; rtol = 1.0e-14, atol = 1.0e-14))
    @test all(isapprox.(CA, CB; rtol = 1.0e-14, atol = 1.0e-14))

    # the production entry agrees with both
    C = itp(GriddedQuery((tx, ty)))
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))

    # cost model sanity: strong upsampling → dim1-first (gather on the SMALL
    # extent); strong downsampling → dim2-first
    @test _gridded_dim2_first(512, 512, 64, 64) == true     # down: gather cost M·N small...
    @test _gridded_dim2_first(64, 64, 512, 512) == false    # up: gather on M·n2=512·64 < M·N·c_g
end

@testitem "NoExtrap validation + unsupported-extrap guards" begin
    using FastInterpolations
    import FastInterpolations as FI

    A = rand(16, 12)
    # NoExtrap: in-domain == point-wise; OOB throws BEFORE any work, naming the axis
    itp = FI.linear_interp((1.0:16.0, 1.0:12.0), A; extrap = FI.NoExtrap())
    tx = collect(range(1.0, 16.0, 21)); ty = collect(range(1.0, 12.0, 17))
    C = itp(GriddedQuery((tx, ty)))
    ref = [itp((x, y)) for x in tx, y in ty]
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))
    @test_throws DomainError itp(GriddedQuery(([0.5, 8.0], ty)))          # axis 1 OOB
    @test_throws DomainError itp(GriddedQuery((tx, [1.0, 12.001])))       # axis 2 OOB
    err = try
        itp(GriddedQuery(([0.5], ty)))
    catch e
        e
    end
    @test err isa DomainError && occursin("axis 1", sprint(showerror, err))

    # Wrap / Fill: explicit ArgumentError at dispatch (both call forms)
    for ex in (FI.WrapExtrap(), FI.FillExtrap(NaN))
        itpu = FI.linear_interp((1.0:16.0, 1.0:12.0), A; extrap = ex)
        @test_throws ArgumentError itpu(GriddedQuery((tx, ty)))
    end
end

@testitem "in-place API: pooled intermediate, zero-alloc, edges" begin
    using FastInterpolations
    import FastInterpolations as FI

    A = rand(48, 40)
    itp = FI.linear_interp((1.0:48.0, 1.0:40.0), A; extrap = FI.ClampExtrap())
    tx = collect(range(1.0, 48.0, 100)); ty = collect(range(1.0, 40.0, 80))
    gq = GriddedQuery((tx, ty))

    # in-place == allocating
    out = Matrix{Float64}(undef, 100, 80)
    ret = itp(out, gq)
    @test ret === out
    @test out == itp(gq)      # same code path + same order → bit-equal

    # wrong size → DimensionMismatch
    @test_throws DimensionMismatch itp(Matrix{Float64}(undef, 99, 80), gq)

    # empty axes: empty output, no pool touched, no throw
    @test size(itp(GriddedQuery((Float64[], ty)))) == (0, 80)
    @test size(itp(GriddedQuery((tx, Float64[])))) == (100, 0)
    e0 = Matrix{Float64}(undef, 0, 80)
    @test itp(e0, GriddedQuery((Float64[], ty))) === e0

    # zero allocation after warmup (pool-backed intermediate; plans are the
    # only remaining O(M+N) allocs — asserted small, then driven to the gate)
    function alloc_count(itp, out, gq)
        itp(out, gq)               # warmup (pool grows here)
        return @allocated itp(out, gq)
    end
    allocs = alloc_count(itp, out, gq)
    # plan vectors (2 Vectors + 2 batch structs) are the only per-call allocs.
    # Measured 3056 B on this configuration (well under the naive O(M+N)
    # formula bound of 4640 B); tightened to measured + 15 %.
    @test allocs <= 3515
end
