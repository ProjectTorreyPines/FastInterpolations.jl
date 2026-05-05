using Pkg
Pkg.activate(".")
using FastInterpolations

grid1 = range(0, 1, length=20)
grid2 = range(0, 1, length=20)
grid3 = range(0, 1, length=20)
data = rand(20, 20, 20)
itp = phs_interp((grid1, grid2, grid3), data; degree=3)

# Directly call
buf1 = zeros(516)
buf2 = zeros(516)
FastInterpolations._phs_solve_stencil!(itp, (2,2,2), buf1, buf2)
println("Cache size: ", length(itp.coeff_caches[1]))
