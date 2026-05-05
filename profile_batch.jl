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

a = @allocated itp(out, queries; deriv=ops_val)
println("First run: ", a)

a2 = @allocated itp(out, queries; deriv=ops_val)
println("Second run: ", a2)
