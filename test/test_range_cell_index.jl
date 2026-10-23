# Cell-index width of range-grid searches. The search truncates and clamps its cell index in
# `Int`, and rebuilds the cell's left edge from the index converted to `_idx_type(T)` first:
# Int32 on a Float32/Float16 grid (GPUs emulate the Int64 → float conversion), `Int` otherwise.
# The returned index is always an `Int`, and the interval is bit-identical to the all-`Int`
# chain.

@testitem "range cell index: 32-bit left-edge conversion on Float32/Float16 grids" setup = [Basic] begin
    using InteractiveUtils: code_llvm
    ir(f, types) = sprint(io -> code_llvm(io, f, types; debuginfo = :none))
    # Integer widths of the float→int truncations and of the int→float conversions in the IR.
    truncations(s) = [m.captures[1] for m in eachmatch(r"fptosi (?:half|float|double) %\S+ to (i\d+)", s)]
    conversions(s) = [m.captures[1] for m in eachmatch(r"sitofp (i\d+) %\S+ to (?:half|float|double)", s)]
    WI = "i$(Sys.WORD_SIZE)"                 # `Int`
    grids(T) = (
        "generic" => FI._to_float(range(zero(T), one(T); length = 101), T),
        "unit" => FI._to_float(3:100, T),
        "unit lo=1" => FI._to_float(1:100, T),
        "OneTo" => FI._to_float(Base.OneTo(100), T),
        "widened" => FI._cached_range(FI._WidenedDomain(), zero(T), one(T), T(1 // 100), T(100), 101),
    )
    for (T, W) in ((Float32, "i32"), (Float16, "i32"), (Float64, WI)), (gname, cr) in grids(T)
        searches = Any[(FI._search_direct_inbounds, Tuple{typeof(cr), T}), (FI._search_direct, Tuple{typeof(cr), T})]
        cr isa FI._CachedRange{<:Any, <:Any, <:FI._AbstractUnitStep} &&
            push!(searches, (FI._search_direct_inbounds, Tuple{typeof(cr), T, typeof(InBounds(last = :exclusive))}))
        for (f, types) in searches
            s = ir(f, types)
            @test !isempty(truncations(s)) && all(==(WI), truncations(s))
            @test !isempty(conversions(s)) && all(==(W), conversions(s))
            @test Base.return_types(f, types) == [Tuple{Int, T, T}]
        end
    end
end

@testitem "range cell index: searches are bit-identical to the all-Int chain" setup = [Basic] begin
    using Random, ForwardDiff
    # The chain the range search has always used (and still uses on Float64 grids).
    function int_chain(cr::FI._CachedRange{T}, q) where {T}
        lo, n = first(cr), length(cr)
        if cr isa FI._CachedRange{<:Any, <:Any, <:FI._AbstractUnitStep}
            i = clamp(unsafe_trunc(Int, lo == one(T) ? q : q - lo + one(T)), 1, n - 1)
            xL = lo == one(T) ? T(i) : lo + (i - 1)
            return i, xL, xL + one(T)
        end
        i = clamp(unsafe_trunc(Int, muladd(q - lo, cr.inv_h, 1)), 1, n - 1)
        xL = muladd(i - 1, cr.h, lo)
        return i, xL, xL + cr.h
    end
    unitstep(cr) = cr isa FI._CachedRange{<:Any, <:Any, <:FI._AbstractUnitStep}
    function check(cr, qs; exclusive = unitstep(cr))
        for q in qs
            ref = int_chain(cr, q)
            @test FI._search_direct(cr, q) === ref
            @test FI._search_direct_inbounds(cr, q) === ref
            exclusive && q < last(cr) && @test FI._search_direct_inbounds(cr, q, InBounds(last = :exclusive)) === ref
        end
        return
    end
    rng = Xoshiro(214)
    grids = (
        FI._to_float(range(0.0f0, 1.0f0; length = 1001), Float32),
        FI._to_float(range(-3.7f0, 12.25f0; length = 333), Float32),
        FI._to_float(range(1.0f6, 1.0f6 + 512; length = 4097), Float32),
        FI._to_float(1:1000, Float32), FI._to_float(-40:900, Float32), FI._to_float(Base.OneTo(777), Float32),
        FI._cached_range(FI._WidenedDomain(), -1.0f0, 1.0f0, 2.0f0 / 200, 100.0f0, 201),
        FI._to_float(range(Float16(0), Float16(1); length = 101), Float16), FI._to_float(1:500, Float16),
    )
    for cr in grids
        T = eltype(cr)
        lo, hi = FI._domain_bounds(cr)
        nodes = [cr[i] for i in eachindex(cr)]
        qs = vcat(lo, hi, nodes, prevfloat.(nodes), nextfloat.(nodes), lo .+ (hi - lo) .* rand(rng, T, 2000))
        qs = filter(q -> lo <= q <= hi, qs)
        check(cr, qs)
        # Genuine Float64 queries (not widened grid-precision ones) and Dual queries
        # (the index follows the primal; the edge stays grid-typed).
        check(cr, filter(q -> lo <= q <= hi, Float64(lo) .+ Float64(hi - lo) .* rand(rng, 2000)))
        for q in qs[1:50:end]
            @test FI._search_direct(cr, ForwardDiff.Dual(q, one(T))) === int_chain(cr, q)
        end
    end
    # At the length limit (searches need no data array). Past 2^24 points a Float32 grid cannot
    # represent its own nodes, which takes the uncapped exclusive-end arm out of its contract
    # (covered separately below), so only the capped searches are compared.
    N = Int(typemax(Int32)) ÷ 2
    for cr in (FI._to_float(range(0.0f0, 1.0f0; length = N), Float32), FI._to_float(1:N, Float32), FI._to_float(-5:(N - 6), Float32))
        lo, hi = first(cr), last(cr)
        check(cr, (lo, hi, prevfloat(hi), nextfloat(lo), cr[N - 1], lo + (hi - lo) / 2, Float64(hi), Float64(hi) - 0.5); exclusive = false)
    end
end

@testitem "range cell index: far out-of-domain queries land on the end cells" setup = [Basic] begin
    grids = (
        FI._to_float(range(0.0f0, 1.0f0; length = 101), Float32),
        FI._to_float(3:100, Float32), FI._to_float(1:100, Float32),
        FI._cached_range(FI._WidenedDomain(), -1.0f0, 1.0f0, 2.0f0 / 200, 100.0f0, 201),
    )
    for cr in grids
        n = length(cr)
        # Finite queries whose index fits `Int` (3f9, 1f10 would overflow Int32): the end cells.
        for q in (-3.0f9, -1.0f10, -3.0e9, prevfloat(first(cr), 4))
            @test FI._search_direct(cr, q)[1] == 1
        end
        for q in (3.0f9, 1.0f10, 3.0e9, nextfloat(last(cr), 4))
            @test FI._search_direct(cr, q)[1] == n - 1
        end
        # Past `Int`'s range `unsafe_trunc` is arbitrary (±Inf, NaN, 1f30): any cell, but a valid
        # one, with the same interval as a query at that cell's midpoint.
        for q in (-Inf32, Inf32, NaN32, NaN, -1.0f30, 1.0f30, -1.0e300, 1.0e300)
            idx, xL, xR = FI._search_direct(cr, q)
            @test 1 <= idx <= n - 1
            @test (idx, xL, xR) === FI._search_direct(cr, (xL + xR) / 2)
        end
    end
    # End to end: ExtendExtrap far outside a Float32 range, 1D and 2D, against Float64.
    x = range(0.0f0, 1.0f0; length = 101)
    y = sin.(3 .* collect(x))
    itp32 = linear_interp(x, y; extrap = ExtendExtrap())
    itp64 = linear_interp(range(0.0, 1.0; length = 101), Float64.(y); extrap = ExtendExtrap())
    for q in (-3.0f9, 3.0f9, -1.0f10, 1.0f10)
        @test itp32(q) ≈ itp64(Float64(q)) rtol = 1.0e-3
    end
    g = (x, range(-1.0f0, 1.0f0; length = 17))
    d = [sin(a) * cos(b) for a in g[1], b in g[2]]
    nd32 = linear_interp(g, d; extrap = ExtendExtrap())
    nd64 = linear_interp(map(r -> range(Float64(first(r)), Float64(last(r)); length = length(r)), g), Float64.(d); extrap = ExtendExtrap())
    @test nd32((3.0f9, 0.25f0)) ≈ nd64((3.0e9, 0.25)) rtol = 1.0e-3
    @test nd32((0.5f0, -3.0f9)) ≈ nd64((0.5, -3.0e9)) rtol = 1.0e-3
end

@testitem "range cell index: Float32/Float16 range grids beyond the Int32 limit are rejected" setup = [Basic] begin
    N = Int(typemax(Int32)) ÷ 2   # half of Int32's range: room for rounding past the last cell
    @test_throws ArgumentError FI._to_float(range(0.0f0, 1.0f0; length = N + 1), Float32)
    @test_throws ArgumentError FI._to_float(1:(N + 1), Float32)
    @test_throws ArgumentError FI._to_float(range(0.0f0, 1.0f0; length = N + 1), Float16)
    @test_throws ArgumentError FI._to_float_adding_endpoint(FI._to_float(range(0.0f0, 1.0f0; length = N), Float32), Float32)
    err = try
        FI._to_float(range(0.0f0, 1.0f0; length = N + 1), Float32)
    catch e
        e
    end
    @test occursin("Float64", sprint(showerror, err))
    # At the limit, and on every other grid type, nothing changes.
    @test length(FI._to_float(range(0.0f0, 1.0f0; length = N), Float32)) == N
    @test length(FI._to_float(range(0.0, 1.0; length = N + 1), Float64)) == N + 1
    @test length(FI._to_float(1:(N + 1), Int)) == N + 1
    # A Float32 grid this long cannot represent its own nodes (`last` rounds up), so the uncapped
    # exclusive-end search may index past the last cell, as it always has; the half-range margin
    # keeps the left edge on the Int chain's value instead of wrapping.
    r = FI._to_float(1000:(1000 + N - 1), Float32)
    idx, xL, _ = FI._search_direct_inbounds(r, Float64(last(r)) - 0.5, InBounds(last = :exclusive))
    @test xL == first(r) + (idx - 1)
end
