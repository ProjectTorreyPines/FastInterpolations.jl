# The grid type-conversion warning was removed as obsolete (one-shot Int grids are
# zero-alloc via the cache; persistent construction allocates a grid copy regardless,
# so "pre-convert for zero-allocation" never applied). Guard against re-introducing a
# spurious per-call warning on the warm Int-grid cubic one-shot — asserted in a fresh
# process since a `maxlog`-throttled warning would be order-dependent in-process.

@testitem "Cubic Int-grid one-shot — no spurious 'allocating conversion' warning" begin
    code = """
    using FastInterpolations
    x = [0, 1, 2, 3, 4, 5, 6, 7]; y = [0, 1, 2, 3, 4, 5]
    data = [sin(1.0a) + cos(1.0b) for a in x, b in y]
    q = (3.4, 2.6)
    for _ in 1:3
        cubic_interp((x, y), data, q)
    end
    """
    err = IOBuffer()
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no --color=no -e $code`
    run(pipeline(cmd; stdout = devnull, stderr = err))
    log = String(take!(err))
    @test !occursin("allocating type conversion", log)
end
