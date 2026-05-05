using Pkg
Pkg.activate(".")
using FastInterpolations

grid1 = range(0, 1, length=20)
grid2 = range(0, 1, length=20)
grid3 = range(0, 1, length=20)
data = rand(20, 20, 20)
itp = phs_interp((grid1, grid2, grid3), data; degree=3)
ops_val = (FastInterpolations.EvalValue(), FastInterpolations.EvalValue(), FastInterpolations.EvalValue())

# warmup
p = (0.5, 0.5, 0.5)
FastInterpolations._phs_eval(itp, p, ops_val)

function profile_inner(itp, p, ops)
    base_idx0 = FastInterpolations._phs_find_base_node(itp, p)
    rhs_buf = zeros(Float64, 60)
    coeff_buf = zeros(Float64, 60)
    
    alloc1 = @allocated FastInterpolations._phs_solve_stencil!(itp, base_idx0, rhs_buf, coeff_buf)
    
    offsets_nb, coeff_nb, hs_nb = FastInterpolations._phs_solve_stencil!(itp, base_idx0, rhs_buf, coeff_buf)
    base_coords = FastInterpolations._phs_base_coords(itp, base_idx0)
    
    alloc2 = @allocated FastInterpolations._phs_eval_coeffs_value(coeff_nb, offsets_nb, hs_nb, p, base_coords, Val(3))
    
    alloc3 = @allocated FastInterpolations._phs_eval_blended(itp, p, ops)
    
    println("Solve: ", alloc1)
    println("Eval Coeffs: ", alloc2)
    println("Blended: ", alloc3)
end

profile_inner(itp, p, ops_val)
