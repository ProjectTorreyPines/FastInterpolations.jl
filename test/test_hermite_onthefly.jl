using Test
using FastInterpolations
using FastInterpolations: _local_slope, PchipSlopes, CardinalSlopes, AkimaSlopes,
    _pchip_slopes!, _cardinal_slopes!, _akima_slopes!,
    _resolve_coeffs,
    _pchip_interp_onthefly, _akima_interp_onthefly, _cardinal_interp_onthefly

@testset "Hermite OnTheFly" begin

    # ========================================
    # 1. Local slope correctness
    # ========================================
    @testset "Local slope == bulk slope" begin
        for n in [2, 3, 4, 5, 10, 50]
            # Uniform grid
            x_u = collect(range(0.0, 1.0, n))
            y_u = sin.(3.0 .* x_u)

            # Non-uniform grid
            x_nu = sort(vcat(0.0, rand(n - 2), 1.0))
            y_nu = sin.(3.0 .* x_nu)

            for (x, y) in [(x_u, y_u), (x_nu, y_nu)]
                # PCHIP
                dy_bulk = similar(y)
                _pchip_slopes!(dy_bulk, x, y)
                for i in 1:n
                    @test _local_slope(PchipSlopes(), x, y, i, n) ≈ dy_bulk[i] atol = 1.0e-14
                end

                # Cardinal (tension=0 and tension=0.5)
                for tension in [0.0, 0.5]
                    dy_c = similar(y)
                    _cardinal_slopes!(dy_c, x, y, tension)
                    for i in 1:n
                        @test _local_slope(CardinalSlopes(tension), x, y, i, n) ≈ dy_c[i] atol = 1.0e-14
                    end
                end

                # Akima
                dy_a = similar(y)
                _akima_slopes!(dy_a, x, y)
                for i in 1:n
                    @test _local_slope(AkimaSlopes(), x, y, i, n) ≈ dy_a[i] atol = 1.0e-14
                end
            end
        end
    end

    # ========================================
    # 2. OnTheFly interpolant == PreCompute
    # ========================================
    @testset "Interpolant: OnTheFly ≈ PreCompute" begin
        x = range(0.0, 2π, 30)
        y = sin.(x)
        xq_pts = range(0.1, 2π - 0.1, 20)

        # PCHIP
        itp_pre = pchip_interp(x, y)
        itp_otf = pchip_interp(x, y; coeffs = OnTheFly())
        for q in xq_pts
            @test itp_otf(q) ≈ itp_pre(q) atol = 1.0e-14
        end

        # Cardinal (tension=0.3)
        itp_c_pre = cardinal_interp(x, y; tension = 0.3)
        itp_c_otf = cardinal_interp(x, y; tension = 0.3, coeffs = OnTheFly())
        for q in xq_pts
            @test itp_c_otf(q) ≈ itp_c_pre(q) atol = 1.0e-14
        end

        # Akima
        itp_a_pre = akima_interp(x, y)
        itp_a_otf = akima_interp(x, y; coeffs = OnTheFly())
        for q in xq_pts
            @test itp_a_otf(q) ≈ itp_a_pre(q) atol = 1.0e-14
        end
    end

    # ========================================
    # 3. OnTheFly oneshot == PreCompute oneshot
    # ========================================
    @testset "Oneshot: OnTheFly ≈ PreCompute" begin
        x = collect(range(0.0, 2π, 25))
        y = sin.(x)
        xq = 1.5

        # Scalar — kwarg API
        @test pchip_interp(x, y, xq; coeffs = OnTheFly()) ≈ pchip_interp(x, y, xq) atol = 1.0e-14
        @test cardinal_interp(x, y, xq; coeffs = OnTheFly()) ≈ cardinal_interp(x, y, xq) atol = 1.0e-14
        @test akima_interp(x, y, xq; coeffs = OnTheFly()) ≈ akima_interp(x, y, xq) atol = 1.0e-14

        # Vector — kwarg API
        xq_vec = collect(range(0.1, 6.0, 15))
        @test pchip_interp(x, y, xq_vec; coeffs = OnTheFly()) ≈ pchip_interp(x, y, xq_vec) atol = 1.0e-14
        @test cardinal_interp(x, y, xq_vec; coeffs = OnTheFly()) ≈ cardinal_interp(x, y, xq_vec) atol = 1.0e-14
        @test akima_interp(x, y, xq_vec; coeffs = OnTheFly()) ≈ akima_interp(x, y, xq_vec) atol = 1.0e-14

        # Cardinal with tension — kwarg API
        @test cardinal_interp(x, y, xq; coeffs = OnTheFly(), tension = 0.5) ≈ cardinal_interp(x, y, xq; tension = 0.5) atol = 1.0e-14

        # Internal API also works
        @test _pchip_interp_onthefly(collect(Float64, x), sin.(x), Float64(xq), NoExtrap(), EvalValue(), AutoSearch(), nothing) ≈ pchip_interp(x, y, xq) atol = 1.0e-14
    end

    # ========================================
    # 4. Derivatives
    # ========================================
    @testset "Derivatives: OnTheFly" begin
        x = range(0.0, 2π, 30)
        y = sin.(x)
        xq = 1.0

        for method in [:pchip, :cardinal, :akima]
            itp_pre = if method == :pchip
                pchip_interp(x, y)
            elseif method == :cardinal
                cardinal_interp(x, y)
            else
                akima_interp(x, y)
            end
            itp_otf = if method == :pchip
                pchip_interp(x, y; coeffs = OnTheFly())
            elseif method == :cardinal
                cardinal_interp(x, y; coeffs = OnTheFly())
            else
                akima_interp(x, y; coeffs = OnTheFly())
            end

            for d in 1:3
                @test itp_otf(xq; deriv = DerivOp(d)) ≈ itp_pre(xq; deriv = DerivOp(d)) atol = 1.0e-13
            end
        end
    end

    # ========================================
    # 5. Extrapolation modes
    # ========================================
    @testset "Extrapolation: OnTheFly" begin
        x = range(0.0, 5.0, 20)
        y = sin.(x)

        for extrap in [ClampExtrap(), FillExtrap(0.0), ExtendExtrap()]
            itp_pre = pchip_interp(x, y; extrap = extrap)
            itp_otf = pchip_interp(x, y; extrap = extrap, coeffs = OnTheFly())
            # In-domain
            @test itp_otf(2.5) ≈ itp_pre(2.5) atol = 1.0e-14
            # Out-of-domain
            @test itp_otf(-0.5) ≈ itp_pre(-0.5) atol = 1.0e-14
            @test itp_otf(5.5) ≈ itp_pre(5.5) atol = 1.0e-14
        end
    end

    # ========================================
    # 6. Type stability
    # ========================================
    @testset "Type stability: @inferred" begin
        x = range(0.0, 2π, 20)
        y = sin.(x)

        # Interpolant
        itp_p = pchip_interp(x, y; coeffs = OnTheFly())
        @test @inferred(itp_p(1.0)) isa Float64

        itp_a = akima_interp(x, y; coeffs = OnTheFly())
        @test @inferred(itp_a(1.0)) isa Float64

        itp_c = cardinal_interp(x, y; coeffs = OnTheFly())
        @test @inferred(itp_c(1.0)) isa Float64

        # Oneshot scalar (internal typed entry — the kwarg entry branches but still infers
        # because coeffs is a compile-time constant)
        xf = collect(Float64, x)
        yf = collect(Float64, y)
        @test @inferred(_pchip_interp_onthefly(xf, yf, 1.0, NoExtrap(), EvalValue(), AutoSearch(), nothing)) isa Float64
        @test @inferred(_akima_interp_onthefly(xf, yf, 1.0, NoExtrap(), EvalValue(), AutoSearch(), nothing)) isa Float64
        @test @inferred(_cardinal_interp_onthefly(xf, yf, 1.0, 0.0, NoExtrap(), EvalValue(), AutoSearch(), nothing)) isa Float64
    end

    # ========================================
    # 7. Edge cases
    # ========================================
    @testset "Edge cases" begin
        # n=2 (linear fallback)
        x2 = [0.0, 1.0]
        y2 = [0.0, 1.0]
        for otf_fn in [
                () -> pchip_interp(x2, y2; coeffs = OnTheFly()),
                () -> cardinal_interp(x2, y2; coeffs = OnTheFly()),
                () -> akima_interp(x2, y2; coeffs = OnTheFly()),
            ]
            itp = otf_fn()
            @test itp(0.5) ≈ 0.5 atol = 1.0e-14
        end

        # n=3
        x3 = [0.0, 0.5, 1.0]
        y3 = [0.0, 1.0, 0.0]
        itp3_pre = pchip_interp(x3, y3)
        itp3_otf = pchip_interp(x3, y3; coeffs = OnTheFly())
        @test itp3_otf(0.25) ≈ itp3_pre(0.25) atol = 1.0e-14

        itp3a_pre = akima_interp(x3, y3)
        itp3a_otf = akima_interp(x3, y3; coeffs = OnTheFly())
        @test itp3a_otf(0.25) ≈ itp3a_pre(0.25) atol = 1.0e-14

        # n=4 (Akima full stencil boundary)
        x4 = [0.0, 0.3, 0.7, 1.0]
        y4 = [0.0, 1.0, 0.5, 0.8]
        itp4_pre = akima_interp(x4, y4)
        itp4_otf = akima_interp(x4, y4; coeffs = OnTheFly())
        for q in [0.1, 0.5, 0.9]
            @test itp4_otf(q) ≈ itp4_pre(q) atol = 1.0e-14
        end
    end

    # ========================================
    # 8. ND via HeteroInterpolantND
    # ========================================
    @testset "ND Hetero: Hermite methods" begin
        x = range(0.0, 2π, 15)
        y = range(0.0, π, 10)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        qx, qy = 1.0, 0.5

        # PchipInterp × LinearInterp
        itp_pl = interp((x, y), data; method = (PchipInterp(), LinearInterp()))
        @test itp_pl((qx, qy)) isa Float64

        # AkimaInterp × CubicInterp
        itp_ac = interp((x, y), data; method = (AkimaInterp(), CubicInterp()))
        @test itp_ac((qx, qy)) isa Float64

        # CardinalInterp × CubicInterp
        itp_cc = interp((x, y), data; method = (CardinalInterp(; tension = 0.3), CubicInterp()))
        @test itp_cc((qx, qy)) isa Float64

        # Verify separable function: PCHIP × Linear
        g(xi) = sin(xi)
        h(yj) = 3.0 * yj + 1.0  # linear in y
        data_sep = [g(xi) * h(yj) for xi in x, yj in y]
        itp_sep = interp((x, y), data_sep; method = (PchipInterp(), LinearInterp()))
        g_ref = pchip_interp(collect(x), [g(xi) for xi in x])(qx)
        h_ref = linear_interp(collect(y), [h(yj) for yj in y])(qy)
        @test itp_sep((qx, qy)) ≈ g_ref * h_ref rtol = 1.0e-10
    end

    # ========================================
    # 9. Homogeneous Hermite ND
    # ========================================
    @testset "ND Homogeneous Hermite" begin
        x = range(0.0, 2π, 15)
        y = range(0.0, π, 10)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        # All-PCHIP
        itp_p = interp((x, y), data; method = PchipInterp())
        @test itp_p((1.0, 0.5)) isa Float64

        # All-Akima
        itp_a = interp((x, y), data; method = AkimaInterp())
        @test itp_a((1.0, 0.5)) isa Float64

        # All-Cardinal
        itp_c = interp((x, y), data; method = CardinalInterp())
        @test itp_c((1.0, 0.5)) isa Float64
    end

    # ========================================
    # 10. CubicInterp + OnTheFly routing
    # ========================================
    @testset "CubicInterp OnTheFly → Hetero" begin
        x = range(0.0, 2π, 15)
        y = range(0.0, π, 10)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp_otf = interp((x, y), data; method = CubicInterp(), coeffs = OnTheFly())
        itp_pre = interp((x, y), data; method = CubicInterp(), coeffs = PreCompute())

        # Should give same results (both cubic spline, just different eval strategy)
        @test itp_otf((1.0, 0.5)) ≈ itp_pre((1.0, 0.5)) rtol = 1.0e-10
    end

    # ========================================
    # 11. AutoCoeffs
    # ========================================
    @testset "AutoCoeffs resolution" begin
        # ── ND overloads ──
        @test _resolve_coeffs(AutoCoeffs(), Val(2), (PchipInterp(), PchipInterp())) isa OnTheFly
        @test _resolve_coeffs(AutoCoeffs(), Val(2), (PchipInterp(), CubicInterp())) isa OnTheFly
        @test _resolve_coeffs(AutoCoeffs(), Val(2), (CubicInterp(), CubicInterp())) isa PreCompute
        @test _resolve_coeffs(AutoCoeffs(), Val(3), (CubicInterp(), CubicInterp(), CubicInterp())) isa OnTheFly

        # ── 1D overloads: scalar, vector, interpolant ──
        xg = collect(range(0.0, 2π, 100))

        # Scalar → always OnTheFly
        @test _resolve_coeffs(AutoCoeffs(), xg, 1.0) isa OnTheFly

        # Vector: few queries → OnTheFly, many queries → PreCompute
        xq_few = collect(range(0.1, 6.0, 10))    # 10 < 100
        xq_many = collect(range(0.1, 6.0, 200))   # 200 > 100
        @test _resolve_coeffs(AutoCoeffs(), xg, xq_few) isa OnTheFly
        @test _resolve_coeffs(AutoCoeffs(), xg, xq_many) isa PreCompute

        # Interpolant (no query) → PreCompute
        @test _resolve_coeffs(AutoCoeffs()) isa PreCompute

        # ── Passthrough ──
        @test _resolve_coeffs(PreCompute(), xg, 1.0) isa PreCompute
        @test _resolve_coeffs(OnTheFly(), xg, xq_many) isa OnTheFly

        # ��─ ND validation ──
        x2d = range(0.0, 2π, 15)
        y2d = range(0.0, π, 10)
        data2d = [sin(xi) * cos(yj) for xi in x2d, yj in y2d]
        @test_throws ArgumentError interp((x2d, y2d), data2d; method = (PchipInterp(), CubicInterp()), coeffs = PreCompute())
        itp_auto = interp((x2d, y2d), data2d; method = (PchipInterp(), CubicInterp()))
        @test itp_auto((1.0, 0.5)) isa Float64

        # ── Verify runtime smart default in oneshot ──
        yg = sin.(xg)

        # Scalar oneshot (default AutoCoeffs → OnTheFly): should match explicit OnTheFly
        @test pchip_interp(xg, yg, 1.0) ≈ pchip_interp(xg, yg, 1.0; coeffs = OnTheFly()) atol = 1.0e-14

        # Vector few (default AutoCoeffs → OnTheFly): should match explicit OnTheFly
        @test pchip_interp(xg, yg, xq_few) ≈ pchip_interp(xg, yg, xq_few; coeffs = OnTheFly()) atol = 1.0e-14

        # Vector many (default AutoCoeffs → PreCompute): should match explicit PreCompute
        @test pchip_interp(xg, yg, xq_many) ≈ pchip_interp(xg, yg, xq_many; coeffs = PreCompute()) atol = 1.0e-14

        # Both strategies produce same results (just different performance)
        @test pchip_interp(xg, yg, xq_many; coeffs = OnTheFly()) ≈ pchip_interp(xg, yg, xq_many; coeffs = PreCompute()) atol = 1.0e-14
    end

    # ========================================
    # 12. AutoCoeffs — type stability + zero allocation
    # ========================================
    @testset "AutoCoeffs type stability" begin
        xg = collect(range(0.0, 2π, 100))
        yg = sin.(xg)

        # _resolve_coeffs itself is type-stable
        @test @inferred(_resolve_coeffs(AutoCoeffs(), xg, 1.0)) isa OnTheFly
        # Vector resolve returns Union{OnTheFly, PreCompute} — runtime branch, expected
        @test _resolve_coeffs(AutoCoeffs(), xg, xg) isa AbstractCoeffStrategy
        @test @inferred(_resolve_coeffs(AutoCoeffs())) isa PreCompute

        # Scalar oneshot with AutoCoeffs default — type-stable end-to-end
        @test @inferred(pchip_interp(xg, yg, 1.0)) isa Float64
        @test @inferred(akima_interp(xg, yg, 1.0)) isa Float64
        @test @inferred(cardinal_interp(xg, yg, 1.0)) isa Float64

        # Internal onthefly paths — type-stable
        @test @inferred(_pchip_interp_onthefly(xg, yg, 1.0, NoExtrap(), EvalValue(), AutoSearch(), nothing)) isa Float64
        @test @inferred(_akima_interp_onthefly(xg, yg, 1.0, NoExtrap(), EvalValue(), AutoSearch(), nothing)) isa Float64
        @test @inferred(_cardinal_interp_onthefly(xg, yg, 1.0, 0.0, NoExtrap(), EvalValue(), AutoSearch(), nothing)) isa Float64
    end

    @testset "AutoCoeffs zero allocation — scalar" begin
        # Function barrier: all setup inside for true allocation measurement
        function _test_auto_scalar_alloc()
            x = collect(range(0.0, 2π, 100))
            y = sin.(x)
            # Warmup
            pchip_interp(x, y, 1.0)
            pchip_interp(x, y, 1.0)
            return @allocated pchip_interp(x, y, 1.0)
        end
        @test _test_auto_scalar_alloc() <= ALLOC_THRESHOLD

        function _test_auto_scalar_alloc_akima()
            x = collect(range(0.0, 2π, 100))
            y = sin.(x)
            akima_interp(x, y, 1.0)
            akima_interp(x, y, 1.0)
            return @allocated akima_interp(x, y, 1.0)
        end
        @test _test_auto_scalar_alloc_akima() <= ALLOC_THRESHOLD

        function _test_auto_scalar_alloc_cardinal()
            x = collect(range(0.0, 2π, 100))
            y = sin.(x)
            cardinal_interp(x, y, 1.0)
            cardinal_interp(x, y, 1.0)
            return @allocated cardinal_interp(x, y, 1.0)
        end
        @test _test_auto_scalar_alloc_cardinal() <= ALLOC_THRESHOLD
    end

    @testset "AutoCoeffs runtime strategy selection" begin
        xg = collect(range(0.0, 2π, 100))
        yg = sin.(xg)

        # ── Verify correct strategy is selected based on xq length ──
        xq_few = collect(range(0.1, 6.0, 10))     # 10 < 100 → OnTheFly
        xq_equal = collect(range(0.1, 6.0, 100))   # 100 == 100 → OnTheFly (≤ threshold)
        xq_many = collect(range(0.1, 6.0, 200))    # 200 > 100 → PreCompute

        @test _resolve_coeffs(AutoCoeffs(), xg, xq_few) isa OnTheFly
        @test _resolve_coeffs(AutoCoeffs(), xg, xq_equal) isa OnTheFly    # == is NOT >
        @test _resolve_coeffs(AutoCoeffs(), xg, xq_many) isa PreCompute

        # ── Both strategies produce identical results ──
        for (method_fn, name) in [(pchip_interp, "pchip"), (akima_interp, "akima"), (cardinal_interp, "cardinal")]
            # Few queries: default=OnTheFly, explicit PreCompute → same result
            v_auto = method_fn(xg, yg, xq_few)
            v_pre = method_fn(xg, yg, xq_few; coeffs = PreCompute())
            v_otf = method_fn(xg, yg, xq_few; coeffs = OnTheFly())
            @test v_auto ≈ v_pre atol = 1.0e-14
            @test v_auto ≈ v_otf atol = 1.0e-14

            # Many queries: default=PreCompute, explicit OnTheFly → same result
            v_auto2 = method_fn(xg, yg, xq_many)
            v_pre2 = method_fn(xg, yg, xq_many; coeffs = PreCompute())
            v_otf2 = method_fn(xg, yg, xq_many; coeffs = OnTheFly())
            @test v_auto2 ≈ v_pre2 atol = 1.0e-14
            @test v_auto2 ≈ v_otf2 atol = 1.0e-14
        end
    end

    # ========================================
    # 13. CS type parameter is explicit
    # ========================================
    @testset "CS type parameter" begin
        x = range(0.0, 2π, 20)
        y = sin.(x)

        itp_pre = pchip_interp(x, y)
        itp_otf = pchip_interp(x, y; coeffs = OnTheFly())

        # CS is the last type parameter
        T_pre = typeof(itp_pre)
        T_otf = typeof(itp_otf)
        @test T_pre.parameters[end] === PreCompute
        @test T_otf.parameters[end] === OnTheFly

        # Same for Akima and Cardinal
        @test typeof(akima_interp(x, y)).parameters[end] === PreCompute
        @test typeof(akima_interp(x, y; coeffs = OnTheFly())).parameters[end] === OnTheFly
        @test typeof(cardinal_interp(x, y)).parameters[end] === PreCompute
        @test typeof(cardinal_interp(x, y; coeffs = OnTheFly())).parameters[end] === OnTheFly
    end

    # ========================================
    # 14. WrapExtrap with OnTheFly
    # ========================================
    @testset "WrapExtrap: OnTheFly" begin
        x = collect(range(0.0, 2π, 30))
        y = sin.(x)
        itp_pre = pchip_interp(x, y; extrap = WrapExtrap())
        itp_otf = pchip_interp(x, y; extrap = WrapExtrap(), coeffs = OnTheFly())

        # In-domain
        @test itp_otf(1.0) ≈ itp_pre(1.0) atol = 1.0e-14
        # Wrap from above
        @test itp_otf(2 * pi + 1.0) ≈ itp_pre(2 * pi + 1.0) atol = 1.0e-14
        # Wrap from below
        @test itp_otf(-1.0) ≈ itp_pre(-1.0) atol = 1.0e-14
        # Boundary exactly at x_max
        @test itp_otf(last(x) - eps()) ≈ itp_pre(last(x) - eps()) atol = 1.0e-14

        # Vector WrapExtrap
        xq_wrap = collect(range(-1.0, 8.0, 20))
        @test pchip_interp(x, y, xq_wrap; extrap = WrapExtrap(), coeffs = OnTheFly()) ≈
            pchip_interp(x, y, xq_wrap; extrap = WrapExtrap()) atol = 1.0e-14
    end

    # ========================================
    # 15. ND gradient correctness with Hermite OnTheFly
    # ========================================
    @testset "ND gradient: Hermite OnTheFly" begin
        xg = range(0.0, 2π, 20)
        yg = range(0.0, π, 15)
        # Separable: f(x,y) = sin(x) * (3y+1) → ∂f/∂x = cos(x)*(3y+1), ∂f/∂y = sin(x)*3
        data = [sin(xi) * (3 * yj + 1) for xi in xg, yj in yg]
        itp = interp((xg, yg), data; method = (PchipInterp(), LinearInterp()))
        qx, qy = 1.0, 0.5
        g = gradient(itp, (qx, qy))
        @test length(g) == 2
        @test g[1] isa Float64
        @test g[2] isa Float64
        # Verify against 1D references
        g_ref = pchip_interp(collect(xg), [sin(xi) for xi in xg])(qx; deriv = DerivOp(1)) * (3 * qy + 1)
        @test g[1] ≈ g_ref rtol = 1.0e-6
    end

    # ========================================
    # 16. Float32 support
    # ========================================
    @testset "Float32: OnTheFly" begin
        x32 = Float32.(collect(range(0.0f0, 2π * 1.0f0, 30)))
        y32 = sin.(x32)

        # Interpolant
        itp_pre = pchip_interp(x32, y32)
        itp_otf = pchip_interp(x32, y32; coeffs = OnTheFly())
        @test itp_otf(1.0f0) isa Float32
        @test itp_otf(1.0f0) ≈ itp_pre(1.0f0) atol = 1.0f-6

        # Akima Float32
        itp_a = akima_interp(x32, y32; coeffs = OnTheFly())
        @test itp_a(1.0f0) isa Float32
        @test itp_a(1.0f0) ≈ akima_interp(x32, y32)(1.0f0) atol = 1.0f-6

        # Cardinal Float32
        itp_c = cardinal_interp(x32, y32; coeffs = OnTheFly())
        @test itp_c(1.0f0) isa Float32
        @test itp_c(1.0f0) ≈ cardinal_interp(x32, y32)(1.0f0) atol = 1.0f-6

        # Scalar oneshot Float32
        v = pchip_interp(x32, y32, 1.0f0; coeffs = OnTheFly())
        @test v isa Float32
    end

end # @testset "Hermite OnTheFly"
