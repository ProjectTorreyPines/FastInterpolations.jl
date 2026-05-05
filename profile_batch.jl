using Pkg
Pkg.activate(".")
using FastInterpolations

grid1 = range(0, 1, length=20)
grid2 = range(0, 1, length=20)
grid3 = range(0, 1, length=20)
data = rand(20, 20, 20)
itp = phs_interp((grid1, grid2, grid3), data; degree=3)

ops_val = (FastInterpolations.EvalValue(), FastInterpolations.EvalValue(), FastInterpolations.EvalValue())
queries = (rand(100), rand(100), rand(100))
out = zeros(100)

function _my_barrier(itp, out, queries, ops, ::Val{N}) where {N}
    nq = length(out)
    Threads.@threads :static for k in 1:nq
        q = FastInterpolations._extract_query_point(queries, k, Val(N))
        oob = FastInterpolations._try_fill_oob(q, itp.grids, itp.extraps, ops, first(itp.data))
        if oob !== nothing
            @inbounds out[k] = oob
        else
            @inbounds out[k] = FastInterpolations._phs_eval(itp, q, ops)
        end
    end
    return out
end

a = @allocated _my_barrier(itp, out, queries, ops_val, Val(3))
println("Barrier first run: ", a)
a2 = @allocated _my_barrier(itp, out, queries, ops_val, Val(3))
println("Barrier second run: ", a2)

