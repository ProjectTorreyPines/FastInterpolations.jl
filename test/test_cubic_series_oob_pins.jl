# Behavior pins for cubic Series OOB semantics (persistent batch vs one-shot batch).
# These freeze the CURRENT contracts so the lean payload-anchor migration
# (docs/design/cubic_series_payload_anchor.md §7/§8) cannot silently drift:
#   * persistent batch OOB formula: `val * one(aq.xq)`      (series_utils.jl)
#   * one-shot batch OOB formula:   `val + zero(xq)*zero(val)` (utils.jl)
# The two differ observably on signed zero — pinned PER SURFACE, on purpose.

@testitem "Clamp OOB signed zero: persistent vs one-shot divergence" begin
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0))          # boundary sample is -0.0
    y1 = vcat(y1, 2.0)
    y2 = collect(range(2.0, 3.0, 11))
    xq = [-0.5, 0.5]                            # OOB-left, in-domain

    @testset "persistent: -0.0 * one(xq) keeps the sign" begin
        sitp = cubic_interp(x, Series(y1, y2); extrap = ClampExtrap())
        outputs = [Vector{Float64}(undef, 2) for _ in 1:2]
        sitp(outputs, xq)
        @test iszero(outputs[1][1])
        @test signbit(outputs[1][1])            # -0.0 preserved

        # deriv: 0 * (-0.0) * one(xq) = -0.0
        sitp(outputs, xq; deriv = DerivOp(1))
        @test iszero(outputs[1][1])
        @test signbit(outputs[1][1])
    end

    @testset "one-shot: -0.0 + zero*zero normalizes to +0.0" begin
        outs = cubic_interp(x, Series(y1, y2), xq; extrap = ClampExtrap())
        @test iszero(outs[1][1])
        @test !signbit(outs[1][1])              # +0.0 (trailing add normalizes)

        outs_d = cubic_interp(x, Series(y1, y2), xq; extrap = ClampExtrap(), deriv = DerivOp(1))
        @test iszero(outs_d[1][1])
        @test !signbit(outs_d[1][1])
    end
end

@testitem "Clamp OOB reads ONLY the boundary sample (NaN neighbors stay masked)" begin
    x = collect(range(0.0, 1.0, 11))
    y_nan = [1.0, 2.0, NaN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    y_ok = collect(range(0.0, 1.0, 11))
    xq = [-0.5, 1.5, 0.05]                      # OOB-left, OOB-right, in-domain (z is NaN-poisoned)

    @testset "persistent" begin
        sitp = cubic_interp(x, Series(y_nan, y_ok); extrap = ClampExtrap())
        outputs = [Vector{Float64}(undef, 3) for _ in 1:2]
        sitp(outputs, xq)
        @test outputs[1][1] == 1.0              # boundary sample, not NaN
        @test outputs[1][2] == 11.0
        @test isnan(outputs[1][3])              # in-domain IS poisoned (z global solve)
    end

    @testset "one-shot" begin
        outs = cubic_interp(x, Series(y_nan, y_ok), xq; extrap = ClampExtrap())
        @test outs[1][1] == 1.0
        @test outs[1][2] == 11.0
        @test isnan(outs[1][3])
    end
end

@testitem "Fill OOB: fill_value propagates (incl. NaN); finite fill deriv is exact zero" begin
    x = collect(range(0.0, 1.0, 11))
    y1 = collect(range(1.0, 2.0, 11))
    y2 = collect(range(2.0, 3.0, 11))
    xq = [-0.5, 0.5]

    @testset "NaN fill_value taints value AND deriv (both surfaces)" begin
        sitp = cubic_interp(x, Series(y1, y2); extrap = FillExtrap(NaN))
        outputs = [Vector{Float64}(undef, 2) for _ in 1:2]
        sitp(outputs, xq)
        @test isnan(outputs[1][1])
        sitp(outputs, xq; deriv = DerivOp(1))
        @test isnan(outputs[1][1])

        outs = cubic_interp(x, Series(y1, y2), xq; extrap = FillExtrap(NaN))
        @test isnan(outs[1][1])
        outs_d = cubic_interp(x, Series(y1, y2), xq; extrap = FillExtrap(NaN), deriv = DerivOp(1))
        @test isnan(outs_d[1][1])
    end

    @testset "finite fill: value passthrough, deriv exact zero (both surfaces)" begin
        sitp = cubic_interp(x, Series(y1, y2); extrap = FillExtrap(7.5))
        outputs = [Vector{Float64}(undef, 2) for _ in 1:2]
        sitp(outputs, xq)
        @test outputs[1][1] === 7.5
        @test outputs[2][1] === 7.5
        sitp(outputs, xq; deriv = DerivOp(1))
        @test outputs[1][1] === 0.0

        outs = cubic_interp(x, Series(y1, y2), xq; extrap = FillExtrap(7.5))
        @test outs[1][1] === 7.5
        outs_d = cubic_interp(x, Series(y1, y2), xq; extrap = FillExtrap(7.5), deriv = DerivOp(1))
        @test outs_d[1][1] === 0.0
    end
end

@testitem "DerivOp(N≥4): 0 * y[idxL] carrier semantics (exact zero / NaN taint)" begin
    x = collect(range(0.0, 1.0, 11))
    y_ok = collect(range(1.0, 2.0, 11))
    y_nan = [1.0, 2.0, NaN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

    # 0.25 lies in cell idx=3 (x[3]=0.2, x[4]=0.3): left tap is the NaN sample.
    @testset "persistent" begin
        sitp = cubic_interp(x, Series(y_ok, y_nan); extrap = ClampExtrap())
        outputs = [Vector{Float64}(undef, 2) for _ in 1:2]
        sitp(outputs, [0.5, 0.25]; deriv = DerivOp(5))
        @test outputs[1][1] === 0.0             # finite data → exact +0.0
        @test isnan(outputs[2][2])              # 0 * NaN taints
    end

    @testset "one-shot" begin
        outs = cubic_interp(x, Series(y_ok, y_nan), [0.5, 0.25]; extrap = ClampExtrap(), deriv = DerivOp(5))
        @test outs[1][1] === 0.0
        @test isnan(outs[2][2])
    end
end

@testitem "Empty query vector: batch entries return untouched empty outputs" begin
    x = collect(range(0.0, 1.0, 11))
    y1 = collect(range(1.0, 2.0, 11))
    y2 = collect(range(2.0, 3.0, 11))

    sitp = cubic_interp(x, Series(y1, y2); extrap = ClampExtrap())
    outputs = [Float64[], Float64[]]
    @test sitp(outputs, Float64[]) === outputs
    @test all(isempty, outputs)

    outs = cubic_interp(x, Series(y1, y2), Float64[]; extrap = ClampExtrap())
    @test all(isempty, outs)
end
