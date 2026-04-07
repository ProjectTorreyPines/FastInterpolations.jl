using Test
using FastInterpolations
using FastInterpolations: _local_slope, PchipSlopes, CardinalSlopes, AkimaSlopes,
    _pchip_slopes!, _cardinal_slopes!, _akima_slopes!,
    _resolve_coeffs, _deriv_size, _is_deriv_method,
    _pchip_interp_onthefly, _akima_interp_onthefly, _cardinal_interp_onthefly,
    _adjoint_func, _throw_onthefly_unsupported

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
            x_nu = sort(vcat(0.0, [0.1 + 0.8 * i / (n - 1) for i in 1:(n - 2)], 1.0))
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
    @testset "Coverage: Hermite method traits" begin
        # _deriv_size: Hermite methods are derivative-based (size=2)
        @test _deriv_size(PchipInterp()) == 2
        @test _deriv_size(CardinalInterp()) == 2
        @test _deriv_size(AkimaInterp()) == 2

        # _is_deriv_method: used by @generated hetero eval kernel
        @test _is_deriv_method(PchipInterp) == true
        @test _is_deriv_method(AkimaInterp) == true
        @test _is_deriv_method(CardinalInterp{Float64}) == true
    end

    @testset "AutoCoeffs resolution" begin
        # ── ND overloads ──
        @test _resolve_coeffs(AutoCoeffs(), Val(2), (PchipInterp(), PchipInterp())) isa OnTheFly
        @test _resolve_coeffs(AutoCoeffs(), Val(2), (PchipInterp(), CubicInterp())) isa OnTheFly
        @test _resolve_coeffs(AutoCoeffs(), Val(2), (CubicInterp(), CubicInterp())) isa PreCompute
        # Global methods (Cubic) stay PreCompute even for N≥3 (integrate/adjoint support)
        @test _resolve_coeffs(AutoCoeffs(), Val(3), (CubicInterp(), CubicInterp(), CubicInterp())) isa PreCompute

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

        # -- ND validation --
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
    # 14. Coverage: PreCompute internal paths (explicit coeffs=PreCompute())
    # ========================================
    @testset "Coverage: explicit PreCompute oneshot" begin
        x = collect(range(0.0, 2π, 25))
        y = sin.(x)
        xq = 1.5
        xq_vec = collect(range(0.1, 6.0, 10))

        # Scalar PreCompute — exercises _*_interp_precompute paths
        @test pchip_interp(x, y, xq; coeffs = PreCompute()) ≈ pchip_interp(x, y, xq; coeffs = OnTheFly()) atol = 1.0e-14
        @test cardinal_interp(x, y, xq; coeffs = PreCompute()) ≈ cardinal_interp(x, y, xq; coeffs = OnTheFly()) atol = 1.0e-14
        @test akima_interp(x, y, xq; coeffs = PreCompute()) ≈ akima_interp(x, y, xq; coeffs = OnTheFly()) atol = 1.0e-14

        # Vector PreCompute
        @test pchip_interp(x, y, xq_vec; coeffs = PreCompute()) ≈ pchip_interp(x, y, xq_vec; coeffs = OnTheFly()) atol = 1.0e-14

        # In-place PreCompute
        out = similar(xq_vec)
        pchip_interp!(out, x, y, xq_vec; coeffs = PreCompute())
        @test out ≈ pchip_interp(x, y, xq_vec; coeffs = OnTheFly()) atol = 1.0e-14
    end

    # ========================================
    # 15. Coverage: show() display
    # ========================================
    @testset "Coverage: show display" begin
        x = range(0.0, 2π, 20)
        y = sin.(x)

        # PreCompute show
        itp_pre = pchip_interp(x, y)
        str_pre = sprint(show, itp_pre)
        @test occursin("monotone", str_pre)
        @test !occursin("on-the-fly", str_pre)

        # OnTheFly show
        itp_otf = pchip_interp(x, y; coeffs = OnTheFly())
        str_otf = sprint(show, itp_otf)
        @test occursin("on-the-fly", str_otf)

        # Detailed show (text/plain)
        str_detail = sprint(show, MIME("text/plain"), itp_otf)
        @test occursin("Coeffs:", str_detail)
        @test occursin("on-the-fly", str_detail)

        # Cardinal + Akima
        str_c = sprint(show, cardinal_interp(x, y; coeffs = OnTheFly()))
        @test occursin("on-the-fly", str_c)
        str_a = sprint(show, akima_interp(x, y; coeffs = OnTheFly()))
        @test occursin("on-the-fly", str_a)

        # Detailed Cardinal + Akima
        str_cd = sprint(show, MIME("text/plain"), cardinal_interp(x, y; coeffs = OnTheFly()))
        @test occursin("Coeffs:", str_cd)
        str_ad = sprint(show, MIME("text/plain"), akima_interp(x, y; coeffs = OnTheFly()))
        @test occursin("Coeffs:", str_ad)
    end

    # ========================================
    # 16. WrapExtrap with OnTheFly
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

    # ========================================
    # 17. Coverage: OnTheFly + ClampExtrap in-domain oneshot (no spacing)
    # ========================================
    @testset "Coverage: OnTheFly ClampExtrap in-domain oneshot" begin
        x = collect(range(0.0, 5.0, 20))
        y = sin.(x)
        mid = 2.5  # clearly inside [0, 5]

        # These exercise the in-domain fallthrough path in _hermite_eval_at_point
        # with sm::AbstractSlopeMethod + _ClampOrFill (hermite_eval.jl lines 279-286)
        for extrap in [ClampExtrap(), FillExtrap(0.0)]
            v_otf = pchip_interp(x, y, mid; coeffs = OnTheFly(), extrap = extrap)
            v_pre = pchip_interp(x, y, mid; coeffs = PreCompute(), extrap = extrap)
            @test v_otf ≈ v_pre atol = 1.0e-14

            v_otf_a = akima_interp(x, y, mid; coeffs = OnTheFly(), extrap = extrap)
            v_pre_a = akima_interp(x, y, mid; coeffs = PreCompute(), extrap = extrap)
            @test v_otf_a ≈ v_pre_a atol = 1.0e-14

            v_otf_c = cardinal_interp(x, y, mid; coeffs = OnTheFly(), extrap = extrap)
            v_pre_c = cardinal_interp(x, y, mid; coeffs = PreCompute(), extrap = extrap)
            @test v_otf_c ≈ v_pre_c atol = 1.0e-14
        end
    end

    # ========================================
    # 18. Coverage: OnTheFly interpolant vector queries (with spacing)
    # ========================================
    @testset "Coverage: OnTheFly interpolant vector eval" begin
        x = range(0.0, 2π, 30)
        y = sin.(collect(x))
        xq_vec = collect(range(0.1, 2π - 0.1, 15))

        for (build_pre, build_otf) in [
                (() -> pchip_interp(x, y), () -> pchip_interp(x, y; coeffs = OnTheFly())),
                (() -> akima_interp(x, y), () -> akima_interp(x, y; coeffs = OnTheFly())),
                (() -> cardinal_interp(x, y), () -> cardinal_interp(x, y; coeffs = OnTheFly())),
            ]
            itp_pre = build_pre()
            itp_otf = build_otf()

            # Vector call on interpolant — exercises interpolant vector loop with spacing + sm
            @test itp_otf(xq_vec) ≈ itp_pre(xq_vec) atol = 1.0e-13

            # In-place variant
            out_pre = similar(xq_vec)
            out_otf = similar(xq_vec)
            itp_pre(out_pre, xq_vec)
            itp_otf(out_otf, xq_vec)
            @test out_otf ≈ out_pre atol = 1.0e-13
        end

        # WrapExtrap interpolant vector — exercises WrapExtrap specialization with spacing + sm
        xq_wrap = collect(range(-1.0, 8.0, 20))
        itp_pre_w = pchip_interp(x, y; extrap = WrapExtrap())
        itp_otf_w = pchip_interp(x, y; extrap = WrapExtrap(), coeffs = OnTheFly())
        @test itp_otf_w(xq_wrap) ≈ itp_pre_w(xq_wrap) atol = 1.0e-13
    end

    # ========================================
    # 19. Coverage: Range + PreCompute in-place (Range disambiguation)
    # ========================================
    @testset "Coverage: Range PreCompute in-place" begin
        x_range = range(0.0, 2π, 30)
        y = sin.(collect(x_range))
        x_query = collect(range(0.1, 6.0, 15))
        out = similar(x_query)

        # These exercise the Range-disambiguated _*_interp_precompute! overloads
        # that require x::AbstractRange + explicit PreCompute
        pchip_interp!(out, x_range, y, x_query; coeffs = PreCompute())
        @test out ≈ pchip_interp(collect(x_range), y, x_query; coeffs = PreCompute()) atol = 1.0e-14

        akima_interp!(out, x_range, y, x_query; coeffs = PreCompute())
        @test out ≈ akima_interp(collect(x_range), y, x_query; coeffs = PreCompute()) atol = 1.0e-14

        cardinal_interp!(out, x_range, y, x_query; coeffs = PreCompute())
        @test out ≈ cardinal_interp(collect(x_range), y, x_query; coeffs = PreCompute()) atol = 1.0e-14
    end

    # ========================================
    # 20. Coverage: QuadraticND OnTheFly path
    # ========================================
    @testset "Coverage: QuadraticND OnTheFly" begin
        xg = range(0.0, 2.0, 15)
        yg = range(0.0, 1.0, 10)
        data = [xi^2 + yj^2 for xi in xg, yj in yg]

        # OnTheFly delegates to HeteroInterpolantND (quadratic_nd_interpolant.jl lines 55-57)
        itp_otf = quadratic_interp((xg, yg), data; coeffs = OnTheFly())
        itp_pre = quadratic_interp((xg, yg), data)
        @test itp_otf((1.0, 0.5)) ≈ itp_pre((1.0, 0.5)) rtol = 1.0e-8
    end

    # ========================================
    # 21. Coverage: _throw_onthefly_unsupported
    # ========================================
    @testset "Coverage: _throw_onthefly_unsupported" begin
        # Direct call
        @test_throws ArgumentError _throw_onthefly_unsupported("integrate")

        # Via integrate on OnTheFly interpolant
        x = range(0.0, 2π, 20)
        y = sin.(collect(x))
        itp_otf = pchip_interp(x, y; coeffs = OnTheFly())
        @test_throws ArgumentError integrate(itp_otf, 0.0, 1.0)
    end

    # ========================================
    # 22. Coverage: _adjoint_func for pchip/akima
    # ========================================
    @testset "Coverage: _adjoint_func pchip/akima" begin
        @test _adjoint_func(pchip_interp) === pchip_adjoint
        @test _adjoint_func(akima_interp) === akima_adjoint
    end

    # ========================================
    # 24. Phase 6: Hint contract under cell-local windowing
    # ========================================
    # Verify the user-facing hint contract is preserved end-to-end through the
    # windowed OnTheFly path:
    #   6a. Hint flow: invalid initial state → updated to absolute indices after first query.
    #   6b. Absolute-coord property: after every query, hint identifies the cell that
    #       actually contains the query in the ORIGINAL grid (not the windowed view).
    #   6c. Batch persistence: hint reuse across monotonic queries doesn't drift.
    @testset "Phase 6: Hint contract under windowing" begin
        x = collect(range(0.0, 2π, 30))
        y = collect(range(-1.0, 1.0, 25))
        z = collect(range(0.0, 1.0, 12))
        data2 = [sin(2xi) * exp(-yj^2) for xi in x, yj in y]
        data3 = [sin(2xi) * exp(-yj^2) * (1 + zk) for xi in x, yj in y, zk in z]

        # ── 6a: Hint flow — invalid initial state ──
        @testset "6a: Invalid initial hint → updated to absolute indices" begin
            for methods in (
                    (PchipInterp(), PchipInterp()),
                    (CardinalInterp(), CardinalInterp()),
                    (AkimaInterp(), AkimaInterp()),
                    (CubicInterp(), CardinalInterp()),
                    (PchipInterp(), LinearInterp()),
                )
                itp = interp((x, y), data2; method = methods, coeffs = OnTheFly())
                hint = (Ref(0), Ref(0))                # deliberately invalid
                v = itp((1.5, 0.4); hint = hint)
                @test v isa Float64
                @test 1 <= hint[1][] <= length(x) - 1   # absolute interval index
                @test 1 <= hint[2][] <= length(y) - 1
                # The cell identified by the hint must contain the query in the original grid.
                @test x[hint[1][]] <= 1.5 <= x[hint[1][] + 1]
                @test y[hint[2][]] <= 0.4 <= y[hint[2][] + 1]
            end
        end

        # ── 6b: Absolute-coord round-trip across many queries ──
        @testset "6b: Absolute-coord property holds after every query" begin
            methods = (CardinalInterp(), CardinalInterp())
            itp = interp((x, y), data2; method = methods, coeffs = OnTheFly())
            hint = (Ref(1), Ref(1))
            # Random scattered queries — exercises walk + reseat behavior
            qs = [(0.5 + 5.5 * rand(), -0.9 + 1.8 * rand()) for _ in 1:100]
            for q in qs
                itp(q; hint = hint)
                # Hint must always be in absolute coords and identify the cell containing q.
                @test 1 <= hint[1][] <= length(x) - 1
                @test 1 <= hint[2][] <= length(y) - 1
                @test x[hint[1][]] <= q[1] <= x[hint[1][] + 1]
                @test y[hint[2][]] <= q[2] <= y[hint[2][] + 1]
            end
        end

        # ── 6b: 3D version ──
        @testset "6b: 3D absolute-coord property" begin
            methods = (PchipInterp(), CardinalInterp(), AkimaInterp())
            itp = interp((x, y, z), data3; method = methods, coeffs = OnTheFly())
            hint = (Ref(1), Ref(1), Ref(1))
            qs = [(0.5 + 5.5 * rand(), -0.9 + 1.8 * rand(), 0.05 + 0.9 * rand()) for _ in 1:50]
            for q in qs
                itp(q; hint = hint)
                @test x[hint[1][]] <= q[1] <= x[hint[1][] + 1]
                @test y[hint[2][]] <= q[2] <= y[hint[2][] + 1]
                @test z[hint[3][]] <= q[3] <= z[hint[3][] + 1]
            end
        end

        # ── 6c: Batch persistence — monotonic sweep ──
        @testset "6c: Batch persistence (monotonic sweep)" begin
            methods = (CardinalInterp(), CardinalInterp())
            itp = interp((x, y), data2; method = methods, coeffs = OnTheFly())
            hint = (Ref(1), Ref(1))
            # Sweep diagonally across the grid; hint should walk monotonically.
            prev_hint_x = 0
            prev_hint_y = 0
            for t in range(0.05, 0.95, 50)
                q = (t * (x[end] - x[1]) + x[1], t * (y[end] - y[1]) + y[1])
                itp(q; hint = hint)
                # Strict monotonic non-decrease (sweep is strictly increasing in both dims).
                @test hint[1][] >= prev_hint_x
                @test hint[2][] >= prev_hint_y
                prev_hint_x = hint[1][]
                prev_hint_y = hint[2][]
                # And still absolute-correct.
                @test x[hint[1][]] <= q[1] <= x[hint[1][] + 1]
                @test y[hint[2][]] <= q[2] <= y[hint[2][] + 1]
            end
        end

        # ── 6: Mixed Cubic × Cardinal — hint persistence on the global-solve axis ──
        @testset "Mixed Cubic×Cardinal: hint contract for both axes" begin
            methods = (CubicInterp(), CardinalInterp())
            itp = interp((x, y), data2; method = methods, coeffs = OnTheFly())
            hint = (Ref(0), Ref(0))
            for q in ((1.5, 0.4), (3.2, -0.6), (5.0, 0.8), (0.3, -0.9))
                itp(q; hint = hint)
                # Cubic axis (global solve): when `_has_any_local_method` is true (Cardinal
                # is local), the entire windowed entry path runs and the pre-search updates
                # ALL real-axis hints — including the Cubic axis. So Cubic-axis hint should
                # also be in absolute coords.
                @test 1 <= hint[1][] <= length(x) - 1
                @test 1 <= hint[2][] <= length(y) - 1
                @test x[hint[1][]] <= q[1] <= x[hint[1][] + 1]
                @test y[hint[2][]] <= q[2] <= y[hint[2][] + 1]
            end
        end
    end

    # ========================================
    # 23. _axis_window per-axis trait (cell-local OnTheFly stencil)
    # ========================================
    @testset "_axis_window trait" begin
        using FastInterpolations: _axis_window, _fixed_window_size

        # ── Fixed window sizes ──
        @test _fixed_window_size(PchipInterp()) == 4
        @test _fixed_window_size(CardinalInterp()) == 4
        @test _fixed_window_size(AkimaInterp()) == 6
        @test _fixed_window_size(LinearInterp()) == 2
        @test _fixed_window_size(ConstantInterp()) == 2

        # ── Cardinal/PCHIP: 4-point window, n=100 ──
        @testset "Cardinal/PCHIP 4-point window" begin
            for m in (CardinalInterp(), PchipInterp())
                # Interior — symmetric
                @test _axis_window(m, 50, 100) == 49:52
                @test _axis_window(m, 25, 100) == 24:27
                # Left boundary — extended right
                @test _axis_window(m, 1, 100) == 1:4
                @test _axis_window(m, 2, 100) == 1:4
                # Right boundary — extended left (cell endpoints are 99, 100; ix=99)
                @test _axis_window(m, 99, 100) == 97:100
                @test _axis_window(m, 98, 100) == 97:100
                # All windows have length == fixed size
                for ix in 1:99
                    @test length(_axis_window(m, ix, 100)) == 4
                end
            end
        end

        # ── Akima: 6-point window, n=100 ──
        @testset "Akima 6-point window" begin
            m = AkimaInterp()
            # Interior — symmetric
            @test _axis_window(m, 50, 100) == 48:53
            @test _axis_window(m, 30, 100) == 28:33
            # Left boundary — extended right
            @test _axis_window(m, 1, 100) == 1:6
            @test _axis_window(m, 2, 100) == 1:6
            @test _axis_window(m, 3, 100) == 1:6
            # Right boundary — extended left
            @test _axis_window(m, 99, 100) == 95:100
            @test _axis_window(m, 97, 100) == 95:100
            # All windows have length == fixed size
            for ix in 1:99
                @test length(_axis_window(m, ix, 100)) == 6
            end
        end

        # ── Linear/Constant: 2-point cell only ──
        @testset "Linear/Constant 2-point cell" begin
            for m in (LinearInterp(), ConstantInterp())
                @test _axis_window(m, 50, 100) == 50:51
                @test _axis_window(m, 1, 100) == 1:2
                @test _axis_window(m, 99, 100) == 99:100
                for ix in 1:99
                    @test length(_axis_window(m, ix, 100)) == 2
                end
            end
        end

        # ── Cubic/Quadratic: full axis (no windowing) ──
        @testset "Cubic/Quadratic full axis" begin
            @test _axis_window(CubicInterp(), 50, 100) == 1:100
            @test _axis_window(QuadraticInterp(), 1, 100) == 1:100
            @test _axis_window(CubicInterp(), 99, 100) == 1:100
        end

        # ── Tiny grid fallback ──
        @testset "Tiny grid fallback (n < fixed_window)" begin
            # Akima needs 6 points; grid of 4 → fall back to full
            @test _axis_window(AkimaInterp(), 1, 4) == 1:4
            @test _axis_window(AkimaInterp(), 2, 4) == 1:4
            @test _axis_window(AkimaInterp(), 3, 4) == 1:4
            # Cardinal needs 4 points; grid of 3 → fall back to full
            @test _axis_window(CardinalInterp(), 1, 3) == 1:3
            @test _axis_window(CardinalInterp(), 2, 3) == 1:3
            # Cardinal grid of exactly 4 → fits
            @test _axis_window(CardinalInterp(), 1, 4) == 1:4
            @test _axis_window(CardinalInterp(), 2, 4) == 1:4
            @test _axis_window(CardinalInterp(), 3, 4) == 1:4
        end

        # ── Window contains both cell endpoints (always) ──
        @testset "Window contains cell endpoints" begin
            for m in (CardinalInterp(), PchipInterp(), AkimaInterp(), LinearInterp(), ConstantInterp())
                for n in (4, 6, 10, 100)
                    n < _fixed_window_size(m) && continue
                    for ix in 1:(n - 1)
                        w = _axis_window(m, ix, n)
                        @test ix in w
                        @test ix + 1 in w
                        @test first(w) >= 1
                        @test last(w) <= n
                    end
                end
            end
        end

        # ── _has_any_local_method (reused from coeff_policy.jl) ──
        @testset "_has_any_local_method gating" begin
            using FastInterpolations: _has_any_local_method
            @test _has_any_local_method((CardinalInterp(), CubicInterp())) == true
            @test _has_any_local_method((CubicInterp(), CardinalInterp())) == true
            @test _has_any_local_method((PchipInterp(), AkimaInterp())) == true
            @test _has_any_local_method((CubicInterp(), CubicInterp())) == false
            @test _has_any_local_method((QuadraticInterp(), CubicInterp())) == false
            @test _has_any_local_method((LinearInterp(), CubicInterp())) == false
        end
    end

end # @testset "Hermite OnTheFly"
