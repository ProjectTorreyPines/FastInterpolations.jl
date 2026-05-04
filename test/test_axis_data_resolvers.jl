@testitem "_resolve_axis — type correctness across (grid, bc) combinations" begin
    using FastInterpolations:
        _resolve_axis, _resolve_data, _resolve_axis_copied,
        _CachedRange, _CachedVector, _ExclusivePeriodicAxis, _ExclusivePeriodicData,
        NoBC, PeriodicBC

    bc_no = NoBC()
    bc_inc = PeriodicBC(endpoint = :inclusive)
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "Vector + non-exclusive → passthrough" begin
        x = [0.0, 1.0, 2.0, 3.0]
        @test _resolve_axis(x, bc_no) === x          # zero-alloc passthrough
        @test _resolve_axis(x, bc_inc) === x          # zero-alloc passthrough
    end

    @testset "Vector + :exclusive → _ExclusivePeriodicAxis" begin
        x = [0.0, 1.0, 2.0, 3.0]
        ax = _resolve_axis(x, bc_excl)
        @test ax isa _ExclusivePeriodicAxis{Float64, Vector{Float64}, Float64}
        @test ax.inner === x                         # zero-copy reference
        @test ax.period == 4.0
        @test length(ax) == 5                        # virtual n+1
    end

    @testset "Range + non-exclusive → _CachedRange (length n)" begin
        r = 0.0:1.0:3.0
        cr = _resolve_axis(r, bc_no)
        @test cr isa _CachedRange{Float64}
        @test length(cr) == 4

        cr_inc = _resolve_axis(r, bc_inc)
        @test cr_inc isa _CachedRange{Float64}
        @test length(cr_inc) == 4
    end

    @testset "Range + :exclusive → _ExclusivePeriodicAxis(_CachedRange, period)" begin
        r = 0.0:1.0:3.0
        cr = _resolve_axis(r, bc_excl)
        @test cr isa FastInterpolations._ExclusivePeriodicAxis
        @test cr.inner isa _CachedRange{Float64}
        @test length(cr) == 5                        # virtual length n+1
        @test length(cr.inner) == 4                  # raw n-length cached Range
        @test last(cr) ≈ 4.0                         # = inner[1] + period
    end
end

@testitem "_resolve_data — type correctness" begin
    using FastInterpolations: _resolve_data, _ExclusivePeriodicData, NoBC, PeriodicBC

    y = [10.0, 20.0, 30.0, 40.0]
    bc_no = NoBC()
    bc_inc = PeriodicBC(endpoint = :inclusive, check = false)
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "non-exclusive → passthrough" begin
        @test _resolve_data(y, bc_no) === y
        @test _resolve_data(y, bc_inc) === y         # check=false skips validation
    end

    @testset ":exclusive → _ExclusivePeriodicData (zero-copy ref)" begin
        yd = _resolve_data(y, bc_excl)
        @test yd isa _ExclusivePeriodicData{Float64, 1, Vector{Float64}}
        @test yd.inner === y                         # zero-copy reference
        @test length(yd) == 5                        # virtual n+1
    end

    @testset ":inclusive endpoint validation when check=true" begin
        # `check` is a type-parameter on `PeriodicBC{E,P,C}` — read via
        # `periodic_check(bc)` (zero-cost). `check=true` is the default.
        bc_inc_check = PeriodicBC(endpoint = :inclusive)   # check=true (default)
        # Bad endpoint raises
        y_bad = [10.0, 20.0, 30.0, 99.0]                    # last ≠ first
        @test_throws ArgumentError _resolve_data(y_bad, bc_inc_check)
        # Good endpoint passes
        y_good = [10.0, 20.0, 30.0, 10.0]
        @test _resolve_data(y_good, bc_inc_check) === y_good
    end
end

@testitem "_resolve_axis_copied — interpolant variant with caching layer" begin
    using FastInterpolations:
        _resolve_axis_copied, _CachedRange, _CachedVector, _ExclusivePeriodicAxis,
        NoBC, PeriodicBC

    bc_no = NoBC()
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

    @testset "Vector + non-exclusive → _CachedVector" begin
        x = [0.0, 1.0, 2.0, 3.0]
        xc = _resolve_axis_copied(x, bc_no, Float64)
        @test xc isa _CachedVector{Float64, Float64}
        @test xc.inner == x                          # copied
        @test xc.inner !== x                         # NOT same object (mutation-safe)
        @test xc.h ≈ [1.0, 1.0, 1.0]                 # cached spacing
    end

    @testset "Vector + :exclusive → _ExclusivePeriodicAxis{_CachedVector inner}" begin
        x = [0.0, 1.0, 2.0, 3.0]
        ax = _resolve_axis_copied(x, bc_excl, Float64)
        @test ax isa _ExclusivePeriodicAxis{Float64, _CachedVector{Float64, Float64}, Float64}
        @test ax.inner isa _CachedVector
        @test ax.inner.h ≈ [1.0, 1.0, 1.0]           # inner is cached
        @test ax.period == 4.0
        @test length(ax) == 5                         # virtual n+1
    end

    @testset "Range + non-exclusive → _CachedRange (length n)" begin
        r = 0.0:1.0:3.0
        cr = _resolve_axis_copied(r, bc_no, Float64)
        @test cr isa _CachedRange{Float64}
        @test length(cr) == 4
    end

    @testset "Range + :exclusive → _ExclusivePeriodicAxis(_CachedRange, period)" begin
        r = 0.0:1.0:3.0
        cr = _resolve_axis_copied(r, bc_excl, Float64)
        @test cr isa FastInterpolations._ExclusivePeriodicAxis
        @test cr.inner isa _CachedRange{Float64}
        @test length(cr) == 5
        @test length(cr.inner) == 4
    end

    @testset "Pre-wrapped re-entry: same-eltype passthrough, different-eltype rebuild" begin
        # `_resolve_axis_copied` on already-wrapped inputs:
        #   - Same eltype as Tg → true passthrough (returns input as-is). The
        #     canonical outer→inner constructor flow re-enters with a wrapper
        #     produced by an earlier `_resolve_axis_copied` call; re-copying
        #     would double allocations. Internal API contract: the wrapper
        #     must already own its inner buffer.
        #   - Different eltype → wrapper-aware `_convert_copy` rebuild
        #     (preserves wrapper type, promotes eltype, single-pass).

        # _CachedVector same-eltype: passthrough
        x_cv = _CachedVector([0.0, 1.0, 2.0])
        out_cv = _resolve_axis_copied(x_cv, bc_no, Float64)
        @test out_cv === x_cv               # true passthrough, no copy

        # _CachedRange: immutable struct, always returned as-is
        x_cr = FastInterpolations._to_float(0.0:1.0:3.0, Float64)
        out_cr = _resolve_axis_copied(x_cr, bc_no, Float64)
        @test out_cr isa _CachedRange{Float64}
        @test out_cr === x_cr               # safe to share — no buffer

        # _ExclusivePeriodicAxis same-eltype: passthrough
        x_ax = _ExclusivePeriodicAxis([0.0, 1.0, 2.0], 3.0)
        out_ax = _resolve_axis_copied(x_ax, bc_excl, Float64)
        @test out_ax === x_ax               # true passthrough

        # Type promotion across wrapper boundary — different eltype rebuilds
        x_cv32 = _CachedVector(Float32[1.0, 2.0, 3.0])
        out_cv32 = _resolve_axis_copied(x_cv32, bc_no, Float64)
        @test out_cv32 isa _CachedVector{Float64, Float64}  # eltype promoted
        @test out_cv32.inner !== x_cv32.inner               # fresh inner buffer

        x_cr32 = FastInterpolations._to_float(0.0f0:1.0f0:3.0f0, Float32)
        out_cr32 = _resolve_axis_copied(x_cr32, bc_no, Float64)
        @test out_cr32 isa _CachedRange{Float64}            # eltype promoted
    end
end

@testitem "Resolvers are type-stable when used inside a function" begin
    using FastInterpolations:
        _resolve_axis, _resolve_data, _resolve_axis_copied,
        _CachedRange, _CachedVector, _ExclusivePeriodicAxis, _ExclusivePeriodicData,
        NoBC, PeriodicBC

    # Each test function takes concrete-typed inputs. We assert the return type
    # via `@inferred` — Julia's compiler must determine the concrete output type
    # purely from input types (no runtime dispatch on `bc`/`x`).

    @testset "_resolve_axis type stability" begin
        x_vec = [0.0, 1.0, 2.0, 3.0]
        x_rng = 0.0:1.0:3.0
        bc_no = NoBC()
        bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

        f_vec_nobc(x, bc) = _resolve_axis(x, bc)
        @test (@inferred f_vec_nobc(x_vec, bc_no)) === x_vec

        f_rng_nobc(x, bc) = _resolve_axis(x, bc)
        @inferred f_rng_nobc(x_rng, bc_no)            # _CachedRange
        @test f_rng_nobc(x_rng, bc_no) isa _CachedRange{Float64}

        f_vec_excl(x, bc) = _resolve_axis(x, bc)
        @inferred f_vec_excl(x_vec, bc_excl)          # _ExclusivePeriodicAxis
        @test f_vec_excl(x_vec, bc_excl) isa _ExclusivePeriodicAxis{Float64}

        f_rng_excl(x, bc) = _resolve_axis(x, bc)
        @inferred f_rng_excl(x_rng, bc_excl)          # _CachedRange of length n+1
        @test length(f_rng_excl(x_rng, bc_excl)) == 5
    end

    @testset "_resolve_data type stability" begin
        y = [10.0, 20.0, 30.0, 40.0]
        bc_no = NoBC()
        bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

        f(y, bc) = _resolve_data(y, bc)
        @test (@inferred f(y, bc_no)) === y
        @inferred f(y, bc_excl)
        @test f(y, bc_excl) isa _ExclusivePeriodicData{Float64, 1, Vector{Float64}}
    end

    @testset "_resolve_axis_copied type stability" begin
        x_vec = [0.0, 1.0, 2.0, 3.0]
        x_rng = 0.0:1.0:3.0
        bc_no = NoBC()
        bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)

        f(x, bc, Tg) = _resolve_axis_copied(x, bc, Tg)
        @inferred f(x_vec, bc_no, Float64)
        @inferred f(x_vec, bc_excl, Float64)
        @inferred f(x_rng, bc_no, Float64)
        @inferred f(x_rng, bc_excl, Float64)
        @test f(x_vec, bc_no, Float64) isa _CachedVector{Float64, Float64}
        @test f(x_vec, bc_excl, Float64) isa
            _ExclusivePeriodicAxis{Float64, _CachedVector{Float64, Float64}}
    end
end

@testitem "Resolvers are zero-alloc on hot path (oneshot use)" setup = [AllocConstants] begin
    using FastInterpolations: _resolve_axis, _resolve_data, NoBC, PeriodicBC

    # ALLOC_THRESHOLD comes from `AllocConstants`: 0 on Julia ≥ 1.12, ~240 B on
    # LTS to absorb infrastructure boxing the LTS compiler can't elide. The
    # surface-API resolvers are designed for zero heap on Julia ≥ 1.12; the
    # struct-allocation cases also accept a small `+ ALLOC_THRESHOLD` slack on
    # LTS for the same reason.
    bc_no = NoBC()
    bc_excl = PeriodicBC(endpoint = :exclusive, period = 4.0)
    x_vec = [0.0, 1.0, 2.0, 3.0]
    y = [10.0, 20.0, 30.0, 40.0]

    # Warmup so JIT compiles the specialized methods
    _resolve_axis(x_vec, bc_no)
    _resolve_axis(x_vec, bc_excl)
    _resolve_data(y, bc_no)
    _resolve_data(y, bc_excl)

    @testset "Vector + NoBC: zero alloc (passthrough)" begin
        @test (@allocated _resolve_axis(x_vec, bc_no)) <= ALLOC_THRESHOLD
        @test (@allocated _resolve_data(y, bc_no)) <= ALLOC_THRESHOLD
    end

    @testset "Vector + :exclusive: ≤ small alloc (one struct)" begin
        # `_ExclusivePeriodicAxis(x, period)` allocates the struct itself
        # (~24-64 bytes on 64-bit Julia) but does NOT copy `x`. Ditto for Data.
        # Acceptable for surface-API one-shot wrappers.
        a_axis = @allocated _resolve_axis(x_vec, bc_excl)
        a_data = @allocated _resolve_data(y, bc_excl)
        @test a_axis <= 64 + ALLOC_THRESHOLD
        @test a_data <= 64 + ALLOC_THRESHOLD
    end

    @testset "Range: stack-allocated _CachedRange (zero heap)" begin
        r = 0.0:1.0:3.0
        _resolve_axis(r, bc_no)       # warmup
        _resolve_axis(r, bc_excl)
        @test (@allocated _resolve_axis(r, bc_no)) <= ALLOC_THRESHOLD
        @test (@allocated _resolve_axis(r, bc_excl)) <= ALLOC_THRESHOLD
    end
end
