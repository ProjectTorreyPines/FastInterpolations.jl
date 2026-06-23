# Pins the hetero promotion-completion contract: coefficient partials are Dual-concrete
# on a Dual grid (Phase 2), and the per-axis output fold keeps Int for all-Constant and
# floats for any dividing axis (Phase 4). See completion spec phases 2 & 4.

@testitem "Hetero partials concrete on Dual grid (Phase 2 coefficient eltype)" begin
    using ForwardDiff: Dual
    g = [Dual{Nothing}(Float64(v), 1.0) for v in 0.0:1.0:5.0]
    data = [Float64(i + 2j) for i in 1:6, j in 1:6]
    # Cubic × Pchip hetero: pchip axis builds slope partials → must be Dual-concrete.
    itp = interp((g, g), data; method = (CubicInterp(), PchipInterp()))
    rt = Base.return_types(itp, Tuple{Float64, Float64})
    @test length(rt) == 1 && isconcretetype(rt[1])
    @test (@inferred itp(2.5, 3.5)) isa Dual
end

@testitem "Hetero partials — Float64 path inferred + zero-alloc (I5)" setup=[AllocConstants] begin
    g = collect(0.0:1.0:5.0)
    data = [Float64(i + 2j) for i in 1:6, j in 1:6]
    itp = interp((g, g), data; method = (CubicInterp(), PchipInterp()))
    @test (@inferred itp(2.5, 3.5)) isa Float64
    h(it, a, b) = (it(a, b); @allocated it(a, b))
    h(itp, 2.5, 3.5)                            # warmup
    @test h(itp, 2.5, 3.5) <= ALLOC_THRESHOLD   # scalar eval pool-bounded
end

@testitem "_hetero_output_eltype — per-axis fold keeps/floats/duals correctly" begin
    using FastInterpolations: _hetero_output_eltype
    C = ConstantInterp(RightSide(), NoBC())   # Int-keeping axis
    L = LinearInterp(NoBC()); K = CubicInterp(NoBC())        # dividing axes
    # all-Constant + all-Int → keeps Int (the corner the legacy form over-floats)
    @test _hetero_output_eltype((C, C), Int, Int, (1, 1)) === Int
    # any dividing axis → floats
    @test _hetero_output_eltype((C, L), Int, Int, (1, 1)) === Float64
    @test _hetero_output_eltype((K, C), Int, Int, (1, 1)) === Float64
    # Float64 everywhere → Float64
    @test _hetero_output_eltype((K, L), Float64, Float64, (1.0, 1.0)) === Float64
end

@testitem "_hetero_output_eltype — Dual rides through; type-level (zero-arg-eval)" begin
    using FastInterpolations: _hetero_output_eltype
    using ForwardDiff: Dual
    DT = Dual{Nothing, Float64, 1}
    L = LinearInterp(NoBC()); K = CubicInterp(NoBC())
    @test _hetero_output_eltype((K, L), DT, Float64, (1.0, 1.0)) <: Dual          # Dual grid
    @test _hetero_output_eltype((K, L), Float64, DT, (1.0, 1.0)) <: Dual          # Dual data
    # Pure type-level: return_types is concrete (the fold constant-folds)
    rt = Base.return_types(_hetero_output_eltype, Tuple{Tuple{typeof(K), typeof(L)}, Type{Float64}, Type{Float64}, Tuple{Float64, Float64}})
    @test length(rt) == 1 && rt[1] === Type{Float64}
end

@testitem "Hetero mixed-method output via _hetero_output_eltype fold (behavior-preserving at generic sites); all-Constant served by homogeneous path" begin
    using ForwardDiff: Dual
    # NOTE: all-Constant hetero is SHORT-CIRCUITED to the homogeneous ConstantInterpolantND path
    # (hetero_interpolant.jl Tuple{ConstantInterp,Vararg{ConstantInterp}} dispatch, and
    # hetero_oneshot.jl lines 236 + 307) BEFORE reaching the 4 wired generic eval sites.
    # Therefore `itp_c(2,3) isa Int` below pins the PUBLIC CONTRACT (homogeneous path honors
    # Int grid/data), NOT the fold at the wired generic sites.
    # The fold's Int-keeping is unit-pinned in Task 5 (`_hetero_output_eltype((C,C),Int,Int,(1,1)) === Int`)
    # and becomes end-to-end observable via the NoInterp path in Task 7.
    # The assertions on `itp_m` (Constant×Linear) ARE served by the generic wired sites and
    # genuinely exercise the fold's behavior-preservation there.

    g = collect(0:1:5)                                  # Int grid
    data = [Int(i + 2j) for i in 1:6, j in 1:6]         # Int data

    # Public-contract pin: all-Constant + Int → keeps Int (via homogeneous short-circuit path).
    itp_c = interp((g, g), data; method = (ConstantInterp(RightSide(), NoBC()),
                                           ConstantInterp(RightSide(), NoBC())))
    @test (@inferred itp_c(2, 3)) isa Int

    # Mixed (Constant×Linear): reaches the 4 wired generic sites — dividing Linear axis floats.
    itp_m = interp((g, g), data; method = (ConstantInterp(RightSide(), NoBC()),
                                           LinearInterp(NoBC())))
    @test (@inferred itp_m(2, 3)) isa Float64

    # Dual-carrier through the generic site: mixed interpolant on a Dual grid must carry Dual
    # through `_hetero_output_eltype` at the wired call sites.
    gd = [Dual{Nothing}(Float64(v), 1.0) for v in 0:1:5]
    data_f = [Float64(i + 2j) for i in 1:6, j in 1:6]
    itp_dual = interp((gd, gd), data_f; method = (ConstantInterp(RightSide(), NoBC()),
                                                    LinearInterp(NoBC())))
    @test (@inferred itp_dual(2.0, 3.0)) isa Dual
    rt = Base.return_types(itp_dual, Tuple{Float64, Float64})
    @test length(rt) == 1 && isconcretetype(rt[1])
end

@testitem "Hetero NoInterp + Constant on real axes — _eval_nointerp @generated path, eval correct" begin
    # NOTE: in HeteroInterpolantND the grid is always float-promoted (Tg=Float64 for Int grids),
    # so _hetero_output_eltype and _promote_eltype both return Float64 for standard numeric types.
    # There is no observable type difference between old and new code here; this test pins the
    # _eval_nointerp OnTheFly path correctness (the 2 wired @generated sites) and type-stability.
    #
    # axis 1: NoInterp (index lookup via GridIdx), axis 2: Constant (real coord)
    # reduced method tuple after NoInterp removal: rm = (ConstantInterp(...),)
    # OnTheFly path (coeffs=OnTheFly()) forces the @generated body's else branch
    # where the 2 wired Tr sites live.
    g = collect(0.0:1.0:5.0)                             # Float64 grid, 6 points
    data = [Float64(i + 2j) for i in 1:6, j in 1:6]     # Float64 data
    itp = interp((g, g), data;
        method = (NoInterp(), ConstantInterp(RightSide(), NoBC())), coeffs = OnTheFly())
    # GridIdx(2) selects row 2 of data; coord 3.0 is within grid 0:1:5
    @test (@inferred itp(GridIdx(2), 3.0)) isa Float64
    # correctness: Constant(RightSide) on axis 2 at coord 3.0
    @test itp(GridIdx(2), 3.0) ≈ constant_interp(g, data[2, :], 3.0; side = RightSide())

    # Dual query on real (Constant) axis — fold must carry Dual through correctly
    using ForwardDiff
    xq_dual = ForwardDiff.Dual{Nothing}(3.0, 1.0)
    val_dual = @inferred itp(GridIdx(2), xq_dual)
    @test val_dual isa ForwardDiff.Dual
    @test ForwardDiff.value(val_dual) ≈ constant_interp(g, data[2, :], 3.0; side = RightSide())
end
