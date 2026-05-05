using Pkg
Pkg.activate(".")
using FastInterpolations
using Profile

grid1 = range(0, 1, length=20)
grid2 = range(0, 1, length=20)
grid3 = range(0, 1, length=20)
data = rand(20, 20, 20)
itp = phs_interp((grid1, grid2, grid3), data; degree=3)

ops_val = (FastInterpolations.EvalValue(), FastInterpolations.EvalValue(), FastInterpolations.EvalValue())

# warmup
rhs_buf = zeros(Float64, 60)
coeff_buf = zeros(Float64, 60)
FastInterpolations._phs_eval_stencil(itp, (10, 10, 10), (0.5, 0.5, 0.5), ops_val, rhs_buf, coeff_buf)

Profile.clear_malloc_data()

for i in 1:1000
    FastInterpolations._phs_eval_stencil(itp, (10, 10, 10), (0.5, 0.5, 0.5), ops_val, rhs_buf, coeff_buf)
end
