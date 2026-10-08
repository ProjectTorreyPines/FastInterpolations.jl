# ND `jacobian` / `jacobian!` / `value_jacobian`: J[i, j] = ∂Fᵢ/∂xⱼ for vector-valued
# data (column j == gradient component j), 1×N for scalar data.

@testsnippet JacobianFixture begin
    using StaticArrays
    const JX = range(0.0, 1.0, 11)
    const JY = range(0.0, 2.0, 21)
    const JZ = range(0.0, 1.5, 9)
    jac_F(a, b) = SA[sin(a) * b, cos(b) * a, a * b]
    # Analytic partials of jac_F: column 1 = ∂/∂a, column 2 = ∂/∂b.
    jac_dF(a, b) = hcat(SA[cos(a) * b, cos(b), b], SA[sin(a), -sin(b) * a, a])
    const JDATA = [jac_F(a, b) for a in JX, b in JY]
    const JQ = (0.37, 1.21)
    const JCTORS = (linear_interp, cubic_interp, quadratic_interp, constant_interp)
end

@testitem "jacobian: columns are the gradient components" setup = [JacobianFixture] begin
    using StaticArrays
    @testset "$(nameof(ctor)), SVector{3} data" for ctor in JCTORS
        itp = ctor((JX, JY), JDATA)
        J = jacobian(itp, JQ)
        g = gradient(itp, JQ)
        @test J isa SMatrix{3, 2, Float64}
        @test J[:, 1] == g[1]
        @test J[:, 2] == g[2]
    end

    @testset "cubic matches the analytic Jacobian" begin
        J = jacobian(cubic_interp((JX, JY), JDATA), JQ)
        @test J ≈ jac_dF(JQ...) atol = 1.0e-4
    end

    @testset "3D cubic → 3×3" begin
        F3(a, b, c) = SA[a * b * c, sin(a) + c^2, b - c]
        itp = cubic_interp((JX, JY, JZ), [F3(a, b, c) for a in JX, b in JY, c in JZ])
        q = (0.37, 1.21, 0.8)
        J = jacobian(itp, q)
        @test J isa SMatrix{3, 3, Float64}
        @test all(J[:, j] == gradient(itp, q)[j] for j in 1:3)
    end
end

@testitem "jacobian: agrees with ForwardDiff on F(q) = itp(q)" setup = [JacobianFixture] begin
    using StaticArrays
    using ForwardDiff
    @testset "$(nameof(ctor))" for ctor in JCTORS
        itp = ctor((JX, JY), JDATA)
        J_ad = ForwardDiff.jacobian(p -> Vector(itp((p[1], p[2]))), collect(JQ))
        @test jacobian(itp, JQ) ≈ J_ad rtol = 1.0e-12 atol = 1.0e-14
    end
end

@testitem "jacobian: shapes for scalar and Vector data" setup = [JacobianFixture] begin
    using StaticArrays
    @testset "scalar data → 1×N row = gradient" begin
        itp = cubic_interp((JX, JY), [sin(a) * b for a in JX, b in JY])
        J = jacobian(itp, JQ)
        @test J isa Matrix{Float64}
        @test size(J) == (1, 2)
        @test vec(J) == collect(gradient(itp, JQ))
    end

    @testset "Vector data → Matrix" begin
        itp = linear_interp((JX, JY), [Vector(jac_F(a, b)) for a in JX, b in JY])
        J = jacobian(itp, JQ)
        @test J isa Matrix{Float64}
        @test size(J) == (3, 2)
        @test J[:, 1] == gradient(itp, JQ)[1]
        @test J[:, 2] == gradient(itp, JQ)[2]
    end

    @testset "matrix-valued data is rejected" begin
        itp = linear_interp((JX, JY), [SA[a b; b a] for a in JX, b in JY])
        @test_throws ArgumentError jacobian(itp, JQ)
        @test_throws ArgumentError value_jacobian(itp, JQ)
        @test_throws ArgumentError jacobian!(zeros(2, 2), itp, JQ)
    end
end

@testitem "jacobian: query forms" setup = [JacobianFixture] begin
    using StaticArrays
    itp = cubic_interp((JX, JY), JDATA)
    J = jacobian(itp, JQ)
    Jv = jacobian(itp, collect(JQ))
    @test Jv isa Matrix{Float64}
    @test Jv == J
    @test jacobian(itp, JQ...) == J
    @test_throws DimensionMismatch jacobian(itp, [0.5])
    @test_throws DimensionMismatch jacobian(itp, [0.5, 0.5, 0.5])
    @test_throws DimensionMismatch jacobian!(zeros(3, 2), itp, [0.5])
    @test_throws DimensionMismatch value_jacobian(itp, [0.5, 0.5, 0.5])
end

@testitem "jacobian: hint is forwarded like gradient" setup = [JacobianFixture] begin
    using StaticArrays
    itp = cubic_interp((JX, JY), JDATA; search = LinearBinarySearch())
    for call in (
            (h) -> jacobian(itp, JQ; hint = h),
            (h) -> jacobian(itp, collect(JQ); hint = h),
            (h) -> value_jacobian(itp, JQ; hint = h),
            (h) -> jacobian!(zeros(3, 2), itp, JQ; hint = h),
        )
        h_ref = (Ref(1), Ref(1))
        gradient(itp, JQ; hint = h_ref)
        h = (Ref(1), Ref(1))
        call(h)
        @test (h[1][], h[2][]) == (h_ref[1][], h_ref[2][])
    end
    @test jacobian(itp, JQ; hint = (Ref(1), Ref(1))) == jacobian(itp, JQ)
end

@testitem "jacobian!: in-place" setup = [JacobianFixture, AllocConstants] begin
    using StaticArrays
    itp = cubic_interp((JX, JY), JDATA)
    J_ref = Matrix(jacobian(itp, JQ))

    J = zeros(3, 2)
    @test jacobian!(J, itp, JQ) === J
    @test J == J_ref
    fill!(J, NaN)
    jacobian!(J, itp, collect(JQ))
    @test J == J_ref
    fill!(J, NaN)
    jacobian!(J, itp, JQ...)
    @test J == J_ref

    @test_throws DimensionMismatch jacobian!(zeros(2, 2), itp, JQ)
    @test_throws DimensionMismatch jacobian!(zeros(3, 3), itp, JQ)

    J32 = zeros(Float32, 3, 2)
    jacobian!(J32, itp, JQ)
    @test J32 == Float32.(J_ref)

    @testset "scalar data → 1×N store" begin
        itps = cubic_interp((JX, JY), [sin(a) * b for a in JX, b in JY])
        J1 = zeros(1, 2)
        jacobian!(J1, itps, JQ)
        @test vec(J1) == collect(gradient(itps, JQ))
        @test_throws DimensionMismatch jacobian!(zeros(2, 2), itps, JQ)
    end

    @testset "Vector data" begin
        itpv = linear_interp((JX, JY), [Vector(jac_F(a, b)) for a in JX, b in JY])
        Jv = zeros(3, 2)
        jacobian!(Jv, itpv, JQ)
        @test Jv == jacobian(itpv, JQ)
    end
end

@testitem "value_jacobian" setup = [JacobianFixture] begin
    using StaticArrays
    itp = cubic_interp((JX, JY), JDATA)
    F, J = value_jacobian(itp, JQ)
    val, g = value_gradient(itp, JQ)
    @test F == val
    @test F == itp(JQ)
    @test J == hcat(g...)                  # exactly the assembled value_gradient
    # `value_gradient` and `gradient` are separate kernels whose SVector results can
    # differ by a few ulps (pre-existing), so `jacobian` is matched approximately.
    @test J ≈ jacobian(itp, JQ) rtol = 1.0e-14
    @test J isa SMatrix{3, 2, Float64}

    Fv, Jv = value_jacobian(itp, collect(JQ))
    @test Fv == F
    @test Jv isa Matrix{Float64}
    @test Jv == J
    @test value_jacobian(itp, JQ...) == (F, J)
end

@testitem "jacobian: inference and allocation" setup = [JacobianFixture, AllocConstants] begin
    using StaticArrays
    itp = cubic_interp((JX, JY), JDATA)
    itps = cubic_interp((JX, JY), [sin(a) * b for a in JX, b in JY])
    q = JQ
    @test (@inferred jacobian(itp, q)) isa SMatrix{3, 2, Float64, 6}
    @test (@inferred value_jacobian(itp, q)) isa Tuple{SVector{3, Float64}, SMatrix{3, 2, Float64, 6}}
    @test (@inferred jacobian(itps, q)) isa Matrix{Float64}

    using Unitful
    itpsu = cubic_interp((JX * u"m", JY * u"m"), [jac_F(a, b) * u"K" for a in JX, b in JY])
    Jsu = @inferred jacobian(itpsu, (JQ[1] * u"m", JQ[2] * u"m"))
    @test isconcretetype(eltype(Jsu))
    @test unit(Jsu[1, 1]) == u"K/m"

    J = zeros(3, 2)
    J1 = zeros(1, 2)
    jacobian(itp, q); value_jacobian(itp, q); jacobian!(J, itp, q); jacobian!(J1, itps, q)
    @test (@allocated jacobian(itp, q)) <= ALLOC_THRESHOLD
    @test (@allocated value_jacobian(itp, q)) <= ALLOC_THRESHOLD
    @test (@allocated jacobian!(J, itp, q)) <= ALLOC_THRESHOLD
    @test (@allocated jacobian!(J1, itps, q)) <= ALLOC_THRESHOLD
end

@testitem "jacobian: Dual query (ForwardDiff through jacobian)" setup = [JacobianFixture] begin
    using StaticArrays
    using ForwardDiff: Dual, value
    D = Dual{Nothing, Float64, 1}
    itp = cubic_interp((JX, JY), JDATA)
    qd = (Dual{Nothing}(JQ[1], 1.0), Dual{Nothing}(JQ[2], 0.0))
    J = @inferred jacobian(itp, qd)
    @test J isa SMatrix{3, 2, D}
    @test value.(J) == jacobian(itp, JQ)
end

@testitem "jacobian: extrap, GridIdx, Hetero, units" setup = [JacobianFixture] begin
    using StaticArrays
    using Unitful
    using ForwardDiff: Dual
    @testset "FillExtrap OOB: the fill value carries the zero" begin
        itpn = cubic_interp((JX, JY), JDATA; extrap = FillExtrap(SA[NaN, NaN, NaN]))
        @test all(isnan, jacobian(itpn, (2.0, 1.0)))
        itpz = cubic_interp((JX, JY), JDATA; extrap = FillExtrap(zero(SVector{3, Float64})))
        @test all(iszero, jacobian(itpz, (2.0, 1.0)))
    end

    @testset "GridIdx axis (generic path)" begin
        itp = cubic_interp((JX, JY), JDATA)
        @test jacobian(itp, (JQ[1], GridIdx(5))) ≈ jacobian(itp, (JQ[1], JY[5])) rtol = 1.0e-10
    end

    @testset "Hetero Cubic × Linear ($(nameof(typeof(coeffs))))" for coeffs in (PreCompute(), OnTheFly())
        itp = interp((JX, JY), JDATA; method = (CubicInterp(), LinearInterp()), coeffs = coeffs)
        J = jacobian(itp, JQ)
        g = gradient(itp, JQ)
        @test J isa SMatrix{3, 2, Float64}
        @test J[:, 1] == g[1] && J[:, 2] == g[2]
    end

    @testset "Hetero with NoInterp (scalar data)" begin
        itp = interp((JX, JY), [sin(a) * b for a in JX, b in JY]; method = (CubicInterp(), NoInterp()))
        q = (JQ[1], GridIdx(5))
        J = jacobian(itp, q)
        @test size(J) == (1, 2)
        @test vec(J) == collect(gradient(itp, q))
        @test J[1, 2] == 0
    end

    @testset "mixed-unit grid: per-column units" begin
        itpu = cubic_interp((JX * u"m", JY * u"s"), [jac_F(a, b) * u"K" for a in JX, b in JY])
        qu = (JQ[1] * u"m", JQ[2] * u"s")
        J = jacobian(itpu, qu)
        g = gradient(itpu, qu)
        @test size(J) == (3, 2)
        @test all(J[:, 1] .== g[1]) && all(J[:, 2] .== g[2])
        @test unit(J[1, 1]) == u"K/m"
        @test unit(J[1, 2]) == u"K/s"

        Jany = Matrix{Any}(undef, 3, 2)
        jacobian!(Jany, itpu, qu)
        @test all(Jany .== J)
        @test_throws ArgumentError jacobian!(zeros(3, 2), itpu, qu)
    end

    @testset "mixed-unit grid × Dual query into a store of the result's entry type" begin
        itpu = cubic_interp((JX * u"m", JY * u"s"), [jac_F(a, b) * u"K" for a in JX, b in JY])
        qd = (Dual{Nothing}(JQ[1], 1.0) * u"m", Dual{Nothing}(JQ[2], 0.0) * u"s")
        Jd = jacobian(itpu, qd)
        store = Matrix{eltype(Jd)}(undef, 3, 2)
        @test jacobian!(store, itpu, qd) === store
        @test all(store .== Jd)
    end
end
