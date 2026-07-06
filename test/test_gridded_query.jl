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
