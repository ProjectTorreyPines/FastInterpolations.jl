# Type stability tests using @inferred
using Test
using FastInterpolations
using FastInterpolations: BCPair, Deriv1, Deriv2, PeriodicBC, NaturalBC, ClampedBC, CubicSplineCache

@testset "Type Stability" begin
    x = collect(range(0.0, 2π, 10))
    y = sin.(x)
    x_query = [0.25, 0.5, 0.75]

    # =========================================================================
    # 2-arg form: CubicInterpolant construction is now type-stable
    # =========================================================================
    @testset "cubic_interp 2-arg @inferred" begin
        # All BC types should be inferrable
        @test @inferred(cubic_interp(x, y)) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=NaturalBC())) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=ClampedBC())) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=PeriodicBC())) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=BCPair(Deriv1(0.0), Deriv2(0.0)))) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=Deriv2(0.0))) isa CubicInterpolant  # PointBC
    end

    @testset "cubic_interp 2-arg extrap variations" begin
        # Different extrap values should all return the SAME type
        itp_none = @inferred cubic_interp(x, y; extrap=:none)
        itp_ext = @inferred cubic_interp(x, y; extrap=:extension)
        itp_const = @inferred cubic_interp(x, y; extrap=:constant)
        itp_wrap = @inferred cubic_interp(x, y; extrap=:wrap)

        # All should be the same concrete type (no E parameter anymore)
        @test typeof(itp_none) === typeof(itp_ext)
        @test typeof(itp_none) === typeof(itp_const)
        @test typeof(itp_none) === typeof(itp_wrap)

        # Type parameters should be {Float64, CubicSplineCache{...}}
        @test itp_none isa CubicInterpolant{Float64}
        @test itp_ext isa CubicInterpolant{Float64}
    end

    @testset "cubic_interp 2-arg BC + extrap combinations" begin
        # Various BC + extrap combinations
        @test @inferred(cubic_interp(x, y; bc=NaturalBC(), extrap=:extension)) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=ClampedBC(), extrap=:constant)) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=BCPair(Deriv1(0.5), Deriv2(-0.5)), extrap=:wrap)) isa CubicInterpolant

        # Periodic BC always uses :wrap internally
        itp_periodic = @inferred cubic_interp(x, y; bc=PeriodicBC())
        @test itp_periodic.extrap === Val(:wrap)
    end

    # =========================================================================
    # 4-arg form: Vector/scalar return types
    # =========================================================================
    @testset "cubic_interp 4-arg Typed BC" begin
        @test @inferred(cubic_interp(x, y, x_query; bc=NaturalBC())) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; bc=ClampedBC())) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; bc=PeriodicBC())) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; bc=BCPair(Deriv1(0.0), Deriv2(0.0)))) isa Vector{Float64}

        # Scalar query
        @test @inferred(cubic_interp(x, y, 0.5; bc=NaturalBC())) isa Float64
        @test @inferred(cubic_interp(x, y, 0.5; bc=ClampedBC())) isa Float64
    end

    @testset "cubic_interp 4-arg extrap variations" begin
        # All extrap values return same Vector type
        @test @inferred(cubic_interp(x, y, x_query; extrap=:none)) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; extrap=:extension)) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; extrap=:constant)) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; extrap=:wrap)) isa Vector{Float64}

        # Scalar
        @test @inferred(cubic_interp(x, y, 0.5; extrap=:extension)) isa Float64
    end

    @testset "cubic_interp! in-place" begin
        out = similar(x_query)
        @test @inferred(cubic_interp!(out, x, y, x_query)) === out
        @test @inferred(cubic_interp!(out, x, y, x_query; bc=ClampedBC())) === out
        @test @inferred(cubic_interp!(out, x, y, x_query; extrap=:extension)) === out
    end

    # =========================================================================
    # Cache-based API
    # =========================================================================
    @testset "Cache-based API" begin
        cache = CubicSplineCache(x; bc=NaturalBC())
        @test @inferred(cubic_interp(cache, y, 0.5; extrap=:none)) isa Float64
        @test @inferred(cubic_interp(cache, y, 0.5; extrap=:extension)) isa Float64

        out = similar(x_query)
        @test @inferred(cubic_interp!(out, cache, y, x_query; extrap=:none)) === out

        # 2-arg with cache
        itp = @inferred cubic_interp(cache, y)
        @test itp isa CubicInterpolant{Float64}
    end

    # =========================================================================
    # CubicInterpolant callable
    # =========================================================================
    @testset "CubicInterpolant callable" begin
        itp = cubic_interp(x, y; bc=NaturalBC())

        # Scalar call
        @test @inferred(itp(0.5)) isa Float64
        @test @inferred(itp(1)) isa Float64  # Int → Float64 conversion

        # Vector call
        @test @inferred(itp(x_query)) isa Vector{Float64}

        # In-place
        out = similar(x_query)
        @test @inferred(itp(out, x_query)) === out
    end

    @testset "CubicInterpolant with different extrap" begin
        # Create interpolants with different extrap modes
        itp_none = cubic_interp(x, y; extrap=:none)
        itp_ext = cubic_interp(x, y; extrap=:extension)

        # Both should be callable with same return types
        @test @inferred(itp_none(0.5)) isa Float64
        @test @inferred(itp_ext(0.5)) isa Float64
        @test @inferred(itp_none(x_query)) isa Vector{Float64}
        @test @inferred(itp_ext(x_query)) isa Vector{Float64}

        # extrap field is different but type is same
        @test itp_none.extrap === Val(:none)
        @test itp_ext.extrap === Val(:extension)
        @test typeof(itp_none) === typeof(itp_ext)
    end

    # =========================================================================
    # Float32 type stability
    # =========================================================================
    @testset "Float32 type stability" begin
        x32 = Float32.(x)
        y32 = Float32.(y)
        xq32 = Float32.(x_query)

        # 2-arg form
        itp32 = @inferred cubic_interp(x32, y32)
        @test itp32 isa CubicInterpolant{Float32}

        # Callable
        @test @inferred(itp32(Float32(0.5))) isa Float32
        @test @inferred(itp32(xq32)) isa Vector{Float32}

        # 4-arg form
        @test @inferred(cubic_interp(x32, y32, xq32)) isa Vector{Float32}
        @test @inferred(cubic_interp(x32, y32, Float32(0.5))) isa Float32
    end

    # =========================================================================
    # Range vs Vector (O(1) vs O(log n) lookup preservation)
    # =========================================================================
    @testset "Range input preserves type" begin
        x_range = range(0.0, 2π, 10)
        y_range = sin.(collect(x_range))

        # Range should be preserved in cache
        itp_range = @inferred cubic_interp(x_range, y_range)
        @test itp_range.cache.x isa AbstractRange

        # Vector input
        itp_vec = @inferred cubic_interp(collect(x_range), y_range)
        @test itp_vec.cache.x isa Vector

        # Both are type-stable and produce same results
        @test @inferred(itp_range(0.5)) ≈ @inferred(itp_vec(0.5))
    end

    # =========================================================================
    # Linear interpolation type stability
    # =========================================================================
    @testset "linear_interp type stability" begin
        @test @inferred(linear_interp(x, y, x_query)) isa Vector{Float64}
        @test @inferred(linear_interp(x, y, 0.5)) isa Float64

        out = similar(x_query)
        @test @inferred(linear_interp!(out, x, y, x_query)) === out

        # LinearInterpolant
        litp = @inferred LinearInterpolant(x, y)
        @test @inferred(litp(0.5)) isa Float64
        @test @inferred(litp(x_query)) isa Vector{Float64}
    end

    # =========================================================================
    # Wrapper functions with Integer grid types
    # Tests that wrapper functions handle Integer → AbstractFloat conversion
    # Note: Direct constructors require AbstractFloat grids
    # =========================================================================
    @testset "Wrapper functions with Integer grid" begin
        @testset "linear_interp with Integer grid" begin
            x_int = 0:10
            y_float = sin.(Float64.(x_int))

            # Wrapper function handles Integer → Float conversion
            litp = @inferred linear_interp(x_int, y_float)
            @test litp isa LinearInterpolant{Float64}
            @test eltype(litp.x) === Float64

            # Integer Vector grid
            x_int_vec = [0, 1, 2, 3, 4]
            y_int_vec = Float64[0, 1, 4, 9, 16]
            litp_vec = @inferred linear_interp(x_int_vec, y_int_vec)
            @test litp_vec isa LinearInterpolant{Float64}

            # Verify interpolation works correctly
            @test litp(5.5) isa Float64
        end

        @testset "constant_interp with Integer grid" begin
            x_int = 1:5
            y_float = Float64[10, 20, 30, 40, 50]

            citp = @inferred constant_interp(x_int, y_float)
            @test citp isa ConstantInterpolant{Float64}
            @test eltype(citp.x) === Float64

            # Integer Vector grid
            x_int_vec = [1, 2, 3, 4, 5]
            citp_vec = @inferred constant_interp(x_int_vec, y_float)
            @test citp_vec isa ConstantInterpolant{Float64}

            # Verify interpolation works correctly
            @test citp(2.5) isa Float64
        end

        @testset "quadratic_interp with Integer grid" begin
            x_int = 0:4
            y_float = Float64.(collect(x_int).^2)  # y = x^2

            qitp = @inferred quadratic_interp(x_int, y_float)
            @test qitp isa QuadraticInterpolant{Float64}
            @test eltype(qitp.x) === Float64

            # Integer Vector grid
            x_int_vec = [0, 1, 2, 3, 4]
            qitp_vec = @inferred quadratic_interp(x_int_vec, y_float)
            @test qitp_vec isa QuadraticInterpolant{Float64}

            # Verify interpolation works correctly
            @test qitp(2.5) isa Float64
        end

        @testset "Mixed Integer x, Integer y via wrappers" begin
            # Both x and y as integers - wrappers handle conversion
            x_int = [1, 2, 3, 4, 5]
            y_int = [10, 20, 30, 40, 50]

            litp = @inferred linear_interp(x_int, y_int)
            @test litp isa LinearInterpolant{Float64}
            @test eltype(litp.x) === Float64
            @test eltype(litp.y) === Float64

            citp = @inferred constant_interp(x_int, y_int)
            @test citp isa ConstantInterpolant{Float64}

            qitp = @inferred quadratic_interp(x_int, y_int)
            @test qitp isa QuadraticInterpolant{Float64}
        end
    end

    # =========================================================================
    # Unified extrap field naming verification
    # All interpolant types should have .extrap field (not .mode)
    # =========================================================================
    @testset "Unified extrap field naming" begin
        @testset "LinearInterpolant extrap field" begin
            litp_none = linear_interp(x, y; extrap=:none)
            litp_const = linear_interp(x, y; extrap=:constant)
            litp_ext = linear_interp(x, y; extrap=:extension)
            litp_wrap = linear_interp(x, y; extrap=:wrap)

            @test litp_none.extrap === Val(:none)
            @test litp_const.extrap === Val(:constant)
            @test litp_ext.extrap === Val(:extension)
            @test litp_wrap.extrap === Val(:wrap)

            # All should be same concrete type
            @test typeof(litp_none) === typeof(litp_const)
            @test typeof(litp_none) === typeof(litp_ext)
            @test typeof(litp_none) === typeof(litp_wrap)
        end

        @testset "ConstantInterpolant extrap field" begin
            citp_none = constant_interp(x, y; extrap=:none)
            citp_const = constant_interp(x, y; extrap=:constant)
            citp_ext = constant_interp(x, y; extrap=:extension)
            citp_wrap = constant_interp(x, y; extrap=:wrap)

            @test citp_none.extrap === Val(:none)
            @test citp_const.extrap === Val(:constant)
            @test citp_ext.extrap === Val(:extension)
            @test citp_wrap.extrap === Val(:wrap)

            # All should be same concrete type
            @test typeof(citp_none) === typeof(citp_const)
            @test typeof(citp_none) === typeof(citp_ext)
            @test typeof(citp_none) === typeof(citp_wrap)
        end

        @testset "QuadraticInterpolant extrap field" begin
            qitp_none = quadratic_interp(x, y; extrap=:none)
            qitp_const = quadratic_interp(x, y; extrap=:constant)
            qitp_ext = quadratic_interp(x, y; extrap=:extension)
            qitp_wrap = quadratic_interp(x, y; extrap=:wrap)

            @test qitp_none.extrap === Val(:none)
            @test qitp_const.extrap === Val(:constant)
            @test qitp_ext.extrap === Val(:extension)
            @test qitp_wrap.extrap === Val(:wrap)

            # All should be same concrete type
            @test typeof(qitp_none) === typeof(qitp_const)
            @test typeof(qitp_none) === typeof(qitp_ext)
            @test typeof(qitp_none) === typeof(qitp_wrap)
        end

        @testset "CubicInterpolant extrap field" begin
            cbitp_none = cubic_interp(x, y; extrap=:none)
            cbitp_const = cubic_interp(x, y; extrap=:constant)
            cbitp_ext = cubic_interp(x, y; extrap=:extension)
            cbitp_wrap = cubic_interp(x, y; extrap=:wrap)

            @test cbitp_none.extrap === Val(:none)
            @test cbitp_const.extrap === Val(:constant)
            @test cbitp_ext.extrap === Val(:extension)
            @test cbitp_wrap.extrap === Val(:wrap)

            # All should be same concrete type
            @test typeof(cbitp_none) === typeof(cbitp_const)
            @test typeof(cbitp_none) === typeof(cbitp_ext)
            @test typeof(cbitp_none) === typeof(cbitp_wrap)
        end
    end

    # =========================================================================
    # ND interpolant type stability
    # =========================================================================

    # Shared 2D test data
    x_nd = range(0.0, 1.0, 11)
    y_nd = range(0.0, 2.0, 15)
    data2d = [sin(xi) * cos(yj) for xi in x_nd, yj in y_nd]

    @testset "ND cubic_interp constructor type correctness" begin
        # Construction works for all extrap modes
        itp_none = cubic_interp((x_nd, y_nd), data2d)
        @test itp_none isa CubicInterpolantND

        itp_const = cubic_interp((x_nd, y_nd), data2d; extrap=:constant)
        @test itp_const isa CubicInterpolantND

        itp_ext = cubic_interp((x_nd, y_nd), data2d; extrap=:extension)
        @test itp_ext isa CubicInterpolantND

        itp_mixed = cubic_interp((x_nd, y_nd), data2d; extrap=(:none, :constant))
        @test itp_mixed isa CubicInterpolantND

        # E type parameter: different extrap → different concrete types
        # (compiler specializes eval per extrap mode, avoiding runtime union-split)
        @test typeof(itp_none) !== typeof(itp_const)
        @test typeof(itp_none) !== typeof(itp_mixed)

        # Eval IS @inferred for all extrap modes (the performance-critical property)
        q = (0.5, 1.0)
        @test @inferred(itp_none(q)) isa Float64
        @test @inferred(itp_const(q)) isa Float64
        @test @inferred(itp_ext(q)) isa Float64
        @test @inferred(itp_mixed(q)) isa Float64
    end

    @testset "ND cubic_interp heterogeneous BC" begin
        # User's real use case: NaturalBC + PeriodicBC with mixed extrap
        x_periodic = range(0.0, 2π, 20)
        y_periodic = range(0.0, 2π, 15)
        data_p = [sin(xi) * cos(yj) for xi in x_periodic, yj in y_periodic]

        itp_mixed_bc = cubic_interp((x_periodic, y_periodic), data_p;
            bc=(NaturalBC(), PeriodicBC()), extrap=(:extension, :wrap))
        @test itp_mixed_bc isa CubicInterpolantND
        @test @inferred(itp_mixed_bc((1.0, 1.0))) isa Float64

        # Homogeneous vs heterogeneous BC must give same type
        itp_homo = cubic_interp((x_nd, y_nd), data2d; bc=NaturalBC())
        itp_hetero = cubic_interp((x_nd, y_nd), data2d; bc=(NaturalBC(), NaturalBC()))
        @test typeof(itp_homo) === typeof(itp_hetero)
    end

    @testset "ND cubic_interp exclusive periodic" begin
        # Exclusive endpoint: data does NOT repeat at boundary
        x_excl = range(0.0, step=0.1, length=10)
        y_excl = range(0.0, step=0.2, length=8)
        data_excl = [sin(2π*xi) * cos(2π*yj) for xi in x_excl, yj in y_excl]

        # Both axes exclusive periodic
        itp_excl = cubic_interp((x_excl, y_excl), data_excl;
            bc=PeriodicBC(; endpoint=:exclusive))
        @test itp_excl isa CubicInterpolantND

        # Mixed: one axis exclusive periodic, other natural
        itp_mixed_excl = cubic_interp((x_excl, y_excl), data_excl;
            bc=(NaturalBC(), PeriodicBC(; endpoint=:exclusive)))
        @test itp_mixed_excl isa CubicInterpolantND

        # Exclusive with explicit per-axis period
        itp_period = cubic_interp((x_excl, y_excl), data_excl;
            bc=(PeriodicBC(; endpoint=:exclusive, period=1.0),
                PeriodicBC(; endpoint=:exclusive, period=1.6)))
        @test itp_period isa CubicInterpolantND

        # Vector grid (non-uniform) with explicit period
        x_vec = collect(x_excl)
        y_vec = collect(y_excl)
        data_vec = [sin(2π*xi) * cos(2π*yj) for xi in x_vec, yj in y_vec]
        itp_vec = cubic_interp((x_vec, y_vec), data_vec;
            bc=(PeriodicBC(; endpoint=:exclusive, period=1.0),
                PeriodicBC(; endpoint=:exclusive, period=1.6)))
        @test itp_vec isa CubicInterpolantND

        # Eval with exclusive periodic should be @inferred
        @test @inferred(itp_excl((0.05, 0.1))) isa Float64
    end

    @testset "ND quadratic_interp constructor type correctness" begin
        itp_none = quadratic_interp((x_nd, y_nd), data2d)
        @test itp_none isa QuadraticInterpolantND

        itp_const = quadratic_interp((x_nd, y_nd), data2d; extrap=:constant)
        @test itp_const isa QuadraticInterpolantND

        itp_mixed = quadratic_interp((x_nd, y_nd), data2d; extrap=(:none, :extension))
        @test itp_mixed isa QuadraticInterpolantND

        # E type parameter: different extrap → different concrete types
        @test typeof(itp_none) !== typeof(itp_const)

        # Eval @inferred
        q = (0.5, 1.0)
        @test @inferred(itp_none(q)) isa Float64
        @test @inferred(itp_const(q)) isa Float64
        @test @inferred(itp_mixed(q)) isa Float64
    end

    @testset "ND linear_interp constructor type correctness" begin
        itp_none = linear_interp((x_nd, y_nd), data2d)
        @test itp_none isa LinearInterpolantND

        itp_const = linear_interp((x_nd, y_nd), data2d; extrap=:constant)
        @test itp_const isa LinearInterpolantND

        itp_mixed = linear_interp((x_nd, y_nd), data2d; extrap=(:none, :wrap))
        @test itp_mixed isa LinearInterpolantND

        # E type parameter: different extrap → different concrete types
        @test typeof(itp_none) !== typeof(itp_const)

        # Eval @inferred
        q = (0.5, 1.0)
        @test @inferred(itp_none(q)) isa Float64
        @test @inferred(itp_const(q)) isa Float64
        @test @inferred(itp_mixed(q)) isa Float64
    end

    @testset "ND constant_interp constructor type correctness" begin
        itp_none = constant_interp((x_nd, y_nd), data2d)
        @test itp_none isa ConstantInterpolantND

        itp_const = constant_interp((x_nd, y_nd), data2d; extrap=:constant)
        @test itp_const isa ConstantInterpolantND

        itp_mixed = constant_interp((x_nd, y_nd), data2d; extrap=(:none, :extension))
        @test itp_mixed isa ConstantInterpolantND

        # E type parameter: different extrap → different concrete types
        @test typeof(itp_none) !== typeof(itp_const)

        # Eval @inferred
        q = (0.5, 1.0)
        @test @inferred(itp_none(q)) isa Float64
        @test @inferred(itp_const(q)) isa Float64
        @test @inferred(itp_mixed(q)) isa Float64
    end

    # =========================================================================
    # Typed AbstractExtrapMode for ND constructors
    # =========================================================================

    @testset "ND typed extrap — cubic_interp" begin
        # All 4 mode types should produce @inferred constructors
        @test @inferred(cubic_interp((x_nd, y_nd), data2d; extrap=NoExtrap())) isa CubicInterpolantND
        @test @inferred(cubic_interp((x_nd, y_nd), data2d; extrap=ConstExtrap())) isa CubicInterpolantND
        @test @inferred(cubic_interp((x_nd, y_nd), data2d; extrap=ExtendExtrap())) isa CubicInterpolantND
        @test @inferred(cubic_interp((x_nd, y_nd), data2d; extrap=WrapExtrap())) isa CubicInterpolantND

        # Per-axis mode tuple
        itp_mixed = @inferred cubic_interp((x_nd, y_nd), data2d; extrap=(NoExtrap(), ConstExtrap()))
        @test itp_mixed isa CubicInterpolantND
        @test itp_mixed.extraps === (NoExtrap(), ConstExtrap())

        # Typed extrap produces identical struct to Symbol extrap
        itp_typed = cubic_interp((x_nd, y_nd), data2d; extrap=NoExtrap())
        itp_sym = cubic_interp((x_nd, y_nd), data2d; extrap=:none)
        @test typeof(itp_typed) === typeof(itp_sym)

        # Eval equivalence
        q = (0.5, 1.0)
        @test @inferred(itp_typed(q)) ≈ itp_sym(q)
    end

    @testset "ND typed extrap — cubic periodic override" begin
        x_p = range(0.0, 2π, 20)
        y_p = range(0.0, 2π, 15)
        data_p = [sin(xi) * cos(yj) for xi in x_p, yj in y_p]

        # PeriodicBC + NoExtrap → axis auto-overridden to WrapExtrap()
        itp = cubic_interp((x_p, y_p), data_p;
            bc=(NaturalBC(), PeriodicBC()), extrap=NoExtrap())
        @test itp.extraps[1] === NoExtrap()
        @test itp.extraps[2] === WrapExtrap()

        # PeriodicBC + ConstExtrap → should throw (incompatible)
        @test_throws ArgumentError cubic_interp((x_p, y_p), data_p;
            bc=(NaturalBC(), PeriodicBC()), extrap=ConstExtrap())
    end

    @testset "ND typed extrap — quadratic_interp" begin
        @test @inferred(quadratic_interp((x_nd, y_nd), data2d; extrap=NoExtrap())) isa QuadraticInterpolantND
        @test @inferred(quadratic_interp((x_nd, y_nd), data2d; extrap=ConstExtrap())) isa QuadraticInterpolantND

        # Per-axis mode tuple
        itp = @inferred quadratic_interp((x_nd, y_nd), data2d; extrap=(NoExtrap(), ExtendExtrap()))
        @test itp.extraps === (NoExtrap(), ExtendExtrap())

        # Typed vs Symbol equivalence
        itp_typed = quadratic_interp((x_nd, y_nd), data2d; extrap=ExtendExtrap())
        itp_sym = quadratic_interp((x_nd, y_nd), data2d; extrap=:extension)
        @test typeof(itp_typed) === typeof(itp_sym)
        @test @inferred(itp_typed((0.5, 1.0))) ≈ itp_sym((0.5, 1.0))
    end

    @testset "ND typed extrap — linear_interp" begin
        @test @inferred(linear_interp((x_nd, y_nd), data2d; extrap=NoExtrap())) isa LinearInterpolantND
        @test @inferred(linear_interp((x_nd, y_nd), data2d; extrap=WrapExtrap())) isa LinearInterpolantND

        # Per-axis mode tuple
        itp = @inferred linear_interp((x_nd, y_nd), data2d; extrap=(ConstExtrap(), WrapExtrap()))
        @test itp.extraps === (ConstExtrap(), WrapExtrap())

        # Typed vs Symbol equivalence
        itp_typed = linear_interp((x_nd, y_nd), data2d; extrap=ConstExtrap())
        itp_sym = linear_interp((x_nd, y_nd), data2d; extrap=:constant)
        @test typeof(itp_typed) === typeof(itp_sym)
        @test @inferred(itp_typed((0.5, 1.0))) ≈ itp_sym((0.5, 1.0))
    end

    @testset "ND typed extrap — constant_interp" begin
        # constant_interp has nested @_dispatch_side_nd, so constructor isn't @inferred
        # (same limitation as Symbol path). Test construction + eval separately.
        itp_none = constant_interp((x_nd, y_nd), data2d; extrap=NoExtrap())
        @test itp_none isa ConstantInterpolantND

        itp_const = constant_interp((x_nd, y_nd), data2d; extrap=ConstExtrap())
        @test itp_const isa ConstantInterpolantND

        itp_ext = constant_interp((x_nd, y_nd), data2d; extrap=ExtendExtrap())
        @test itp_ext isa ConstantInterpolantND

        itp_wrap = constant_interp((x_nd, y_nd), data2d; extrap=WrapExtrap())
        @test itp_wrap isa ConstantInterpolantND

        # Per-axis mode tuple
        itp = constant_interp((x_nd, y_nd), data2d; extrap=(NoExtrap(), ConstExtrap()))
        @test itp.extraps === (NoExtrap(), ConstExtrap())

        # Typed vs Symbol equivalence
        itp_typed = constant_interp((x_nd, y_nd), data2d; extrap=NoExtrap())
        itp_sym = constant_interp((x_nd, y_nd), data2d; extrap=:none)
        @test typeof(itp_typed) === typeof(itp_sym)
        @test @inferred(itp_typed((0.5, 1.0))) ≈ itp_sym((0.5, 1.0))
    end

    @testset "ND oneshot scalar @inferred" begin
        q = (0.5, 1.0)

        @test @inferred(cubic_interp((x_nd, y_nd), data2d, q)) isa Float64
        @test @inferred(quadratic_interp((x_nd, y_nd), data2d, q)) isa Float64
        @test @inferred(linear_interp((x_nd, y_nd), data2d, q)) isa Float64
        @test @inferred(constant_interp((x_nd, y_nd), data2d, q)) isa Float64
    end

    # =========================================================================
    # Typed AbstractExtrapMode for ND oneshot functions
    # =========================================================================

    @testset "ND oneshot typed extrap — scalar @inferred" begin
        q = (0.5, 1.0)

        # All 4 algorithms with typed extrap
        @test @inferred(cubic_interp((x_nd, y_nd), data2d, q; extrap=NoExtrap())) isa Float64
        @test @inferred(cubic_interp((x_nd, y_nd), data2d, q; extrap=ConstExtrap())) isa Float64
        @test @inferred(quadratic_interp((x_nd, y_nd), data2d, q; extrap=NoExtrap())) isa Float64
        @test @inferred(quadratic_interp((x_nd, y_nd), data2d, q; extrap=ExtendExtrap())) isa Float64
        @test @inferred(linear_interp((x_nd, y_nd), data2d, q; extrap=NoExtrap())) isa Float64
        @test @inferred(linear_interp((x_nd, y_nd), data2d, q; extrap=WrapExtrap())) isa Float64

        # constant_interp: eval is @inferred even though constructor isn't
        @test @inferred(constant_interp((x_nd, y_nd), data2d, q; extrap=NoExtrap())) isa Float64
    end

    @testset "ND oneshot typed extrap — equivalence with Symbol" begin
        q = (0.5, 1.0)

        # Typed vs Symbol should produce identical results
        @test cubic_interp((x_nd, y_nd), data2d, q; extrap=NoExtrap()) ≈
              cubic_interp((x_nd, y_nd), data2d, q; extrap=:none)
        @test cubic_interp((x_nd, y_nd), data2d, q; extrap=ConstExtrap()) ≈
              cubic_interp((x_nd, y_nd), data2d, q; extrap=:constant)

        @test quadratic_interp((x_nd, y_nd), data2d, q; extrap=ExtendExtrap()) ≈
              quadratic_interp((x_nd, y_nd), data2d, q; extrap=:extension)

        @test linear_interp((x_nd, y_nd), data2d, q; extrap=WrapExtrap()) ≈
              linear_interp((x_nd, y_nd), data2d, q; extrap=:wrap)

        @test constant_interp((x_nd, y_nd), data2d, q; extrap=ConstExtrap()) ≈
              constant_interp((x_nd, y_nd), data2d, q; extrap=:constant)
    end

    @testset "ND oneshot typed extrap — per-axis mode tuple" begin
        q = (0.5, 1.0)

        # Per-axis typed mode tuple
        @test @inferred(cubic_interp((x_nd, y_nd), data2d, q;
            extrap=(NoExtrap(), ConstExtrap()))) isa Float64
        @test @inferred(linear_interp((x_nd, y_nd), data2d, q;
            extrap=(ConstExtrap(), WrapExtrap()))) isa Float64
    end

    @testset "ND oneshot typed extrap — batch SoA" begin
        qs = (collect(range(0.2, 0.8, 5)), collect(range(0.5, 1.5, 5)))

        # SoA batch with typed extrap
        result_typed = cubic_interp((x_nd, y_nd), data2d, qs; extrap=NoExtrap())
        result_sym = cubic_interp((x_nd, y_nd), data2d, qs; extrap=:none)
        @test result_typed ≈ result_sym

        result_typed = linear_interp((x_nd, y_nd), data2d, qs; extrap=ConstExtrap())
        result_sym = linear_interp((x_nd, y_nd), data2d, qs; extrap=:constant)
        @test result_typed ≈ result_sym
    end

    # =========================================================================
    # PeriodicBC + Mode type: type stability for constructors and oneshot
    # =========================================================================

    @testset "ND PeriodicBC + Mode — constructor type stability" begin
        x_p = range(0.0, 2π, 21)
        y_p = range(0.0, 2π, 21)
        data_p = [sin(xi) * cos(yj) for xi in x_p, yj in y_p]

        # Both axes periodic + NoExtrap (auto-overrides to wrap)
        itp = @inferred cubic_interp((x_p, y_p), data_p;
            bc=PeriodicBC(), extrap=NoExtrap())
        @test itp.extraps === (WrapExtrap(), WrapExtrap())

        # Mixed: one periodic, one natural
        itp_mixed = @inferred cubic_interp((x_p, y_p), data_p;
            bc=(NaturalBC(), PeriodicBC()), extrap=NoExtrap())
        @test itp_mixed.extraps[1] === NoExtrap()
        @test itp_mixed.extraps[2] === WrapExtrap()

        # Quadratic: PeriodicBC + WrapExtrap
        itp_q = @inferred quadratic_interp((x_p, y_p), data_p;
            bc=NaturalBC(), extrap=WrapExtrap())
        @test itp_q.extraps === (WrapExtrap(), WrapExtrap())
    end

    @testset "ND PeriodicBC exclusive + Mode — constructor type stability" begin
        x_excl = range(0.0, step=0.1, length=20)
        y_excl = range(0.0, step=0.2, length=10)
        data_excl = [sin(2π*xi) * cos(2π*yj) for xi in x_excl, yj in y_excl]

        # Exclusive periodic + WrapExtrap
        itp = @inferred cubic_interp((x_excl, y_excl), data_excl;
            bc=PeriodicBC(; endpoint=:exclusive), extrap=WrapExtrap())
        @test itp isa CubicInterpolantND
        @test @inferred(itp((0.05, 0.1))) isa Float64

        # Mixed: exclusive periodic axis + natural axis + NoExtrap
        itp_mixed = @inferred cubic_interp((x_excl, y_excl), data_excl;
            bc=(NaturalBC(), PeriodicBC(; endpoint=:exclusive)),
            extrap=NoExtrap())
        @test itp_mixed isa CubicInterpolantND
        @test itp_mixed.extraps[1] === NoExtrap()
        @test itp_mixed.extraps[2] === WrapExtrap()

        # Exclusive with explicit period + Mode
        itp_period = @inferred cubic_interp((x_excl, y_excl), data_excl;
            bc=(PeriodicBC(; endpoint=:exclusive, period=2.0),
                PeriodicBC(; endpoint=:exclusive, period=2.0)),
            extrap=WrapExtrap())
        @test itp_period isa CubicInterpolantND
    end

    @testset "ND PeriodicBC + Mode — oneshot type stability" begin
        x_p = range(0.0, 2π, 21)
        y_p = range(0.0, 2π, 21)
        data_p = [sin(xi) * cos(yj) for xi in x_p, yj in y_p]
        q = (1.5, 0.8)

        # Cubic: PeriodicBC + WrapExtrap oneshot
        @test @inferred(cubic_interp((x_p, y_p), data_p, q;
            bc=PeriodicBC(), extrap=WrapExtrap())) isa Float64

        # Cubic: mixed BC + NoExtrap oneshot (periodic auto-override)
        @test @inferred(cubic_interp((x_p, y_p), data_p, q;
            bc=(NaturalBC(), PeriodicBC()), extrap=NoExtrap())) isa Float64

        # Quadratic: PeriodicBC + WrapExtrap oneshot
        @test @inferred(quadratic_interp((x_p, y_p), data_p, q;
            bc=NaturalBC(), extrap=WrapExtrap())) isa Float64

        # Linear: WrapExtrap oneshot (no BC for linear)
        @test @inferred(linear_interp((x_p, y_p), data_p, q;
            extrap=WrapExtrap())) isa Float64

        # Constant: WrapExtrap oneshot (no BC for constant)
        @test @inferred(constant_interp((x_p, y_p), data_p, q;
            extrap=WrapExtrap())) isa Float64
    end

    @testset "ND PeriodicBC exclusive + Mode — oneshot type stability" begin
        x_excl = range(0.0, step=0.1, length=20)
        y_excl = range(0.0, step=0.2, length=10)
        data_excl = [sin(2π*xi) * cos(2π*yj) for xi in x_excl, yj in y_excl]
        q = (0.5, 0.5)

        # Cubic: exclusive periodic + WrapExtrap oneshot
        @test @inferred(cubic_interp((x_excl, y_excl), data_excl, q;
            bc=PeriodicBC(; endpoint=:exclusive), extrap=WrapExtrap())) isa Float64

        # Cubic: mixed BC exclusive + NoExtrap oneshot
        @test @inferred(cubic_interp((x_excl, y_excl), data_excl, q;
            bc=(NaturalBC(), PeriodicBC(; endpoint=:exclusive)),
            extrap=NoExtrap())) isa Float64
    end
end
