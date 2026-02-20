using FastInterpolations
using BenchmarkTools

function run_integral_bench()
    x = collect(range(0.0, 1.0, length=1001))
    y = @. sin(2pi*x) + 0.3x

    println("=== 1D Cubic Integration ===")
    itp_cub = cubic_interp(x, y; extrap=NoExtrap())
    print("  full-domain:  "); @btime integrate($itp_cub)
    print("  sub-range:    "); @btime integrate($itp_cub, 0.13, 0.87)

    println("\n=== 1D Linear Integration ===")
    itp_lin = linear_interp(x, y; extrap=NoExtrap())
    print("  full-domain:  "); @btime integrate($itp_lin)
    print("  sub-range:    "); @btime integrate($itp_lin, 0.13, 0.87)

    println("\n=== 1D Quadratic Integration ===")
    itp_quad = quadratic_interp(x, y; extrap=NoExtrap())
    print("  full-domain:  "); @btime integrate($itp_quad)
    print("  sub-range:    "); @btime integrate($itp_quad, 0.13, 0.87)

    println("\n=== Extrap modes (cubic, sub-range) ===")
    itp_const = cubic_interp(x, y; extrap=ConstExtrap())
    itp_wrap = cubic_interp(x, y; extrap=WrapExtrap())
    print("  NoExtrap   "); @btime integrate($itp_cub, 0.13, 0.87)
    print("  ConstExtrap"); @btime integrate($itp_const, -0.2, 1.2)
    print("  WrapExtrap "); @btime integrate($itp_wrap, -0.2, 2.8)

    println("\n=== 2D Cubic Integration ===")
    xg = collect(range(0.0, 1.0, length=51))
    yg = collect(range(0.0, 1.0, length=51))
    data = [sin(2pi*xi) * cos(2pi*yj) for xi in xg, yj in yg]
    itp_2d = cubic_interp((xg, yg), data; extrap=(:none, :none))
    print("  full-domain:  "); @btime integrate($itp_2d)
    print("  sub-range:    "); @btime integrate($itp_2d, (0.1, 0.1), (0.9, 0.9))
end

run_integral_bench()
