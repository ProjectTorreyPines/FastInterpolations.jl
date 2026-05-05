using Pkg
Pkg.activate(".")
using FastInterpolations
using InteractiveUtils

grid1 = range(0, 1, length=20)
grid2 = range(0, 1, length=20)
grid3 = range(0, 1, length=20)
data = rand(20, 20, 20)
itp = phs_interp((grid1, grid2, grid3), data; degree=3)
ops_val = (FastInterpolations.EvalValue(), FastInterpolations.EvalValue(), FastInterpolations.EvalValue())

# print to file to avoid truncation
open("warntype.txt", "w") do io
    code_warntype(io, FastInterpolations._phs_eval_blended, (typeof(itp), typeof((0.5, 0.5, 0.5)), typeof(ops_val)))
end
