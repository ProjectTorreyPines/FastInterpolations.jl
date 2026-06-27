# The warm Int-Vector-grid cubic one-shot is zero-alloc, yet currently emits a
# `@warn "...allocating type conversion..."` from the internal pooled cubic rebuild
# on the windowed Int view — wrongly telling users to pre-convert for zero-alloc.
# `maxlog=1` makes in-process log capture order-dependent, so assert in a fresh process.

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
