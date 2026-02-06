using FastInterpolations 	
using Plots
using Printf
using LinearAlgebra

f_point(xi, yi) = sin(2π * xi) * cos(2π * yi)

function plot_comparison(xs, ys)
    f_test(x, y) = [f_point(xi, yj) for xi in x, yj in y]

    itp_constant = constant_interp((xs, ys), f_test(xs, ys))
    itp_linear = linear_interp((xs, ys), f_test(xs, ys))
    itp_cubic = cubic_interp((xs, ys), f_test(xs, ys); bc=(PeriodicBC(), PeriodicBC()))
    # itp_cubic = cubic_interp((xs, ys), f_test(xs, ys))

    # High-res grid for ground truth 
    x_hi = collect(range(minimum(xs),maximum(xs),length=400))
    y_hi = collect(range(minimum(ys),maximum(ys),length=400))
    z_hi = f_test(x_hi, y_hi)

    # CMAP = :viridis
    CMAP = :RdBu
    # CMAP = :seismic
    CRANGE = 0.9.*extrema(z_hi)
    kwargs = (c=CMAP, clims=CRANGE, aspect_ratio=:equal, xlabel="x", ylabel="y", xticks=0:0.2:1, yticks=0:0.2:1,
            xlims=extrema(xs), ylims=extrema(ys),  titlefont=(14, "Helvetica Bold"), topmargin=9Plots.px, legend=false)

    itp_plot_kwargs = (node_color=:black, node_size=6, node_alpha=0.4, gridline_alpha=0.3)

    # Calculate Error
    function measure_error_string(itp)
        z_interp = [itp((xi, yi)) for xi in x_hi, yi in y_hi]
        return @sprintf("%.1f %%", 100*(norm(z_interp .- z_hi)/norm(z_hi) ))
    end


    # Create plots
    p1 = heatmap(x_hi, y_hi, z_hi', title="Ground Truth"; kwargs..., titlefont=(17, "Helvetica Bold"))
    p2 = plot(itp_constant; title="Constant Interpolation\n (Error≈$(measure_error_string(itp_constant)))", kwargs..., itp_plot_kwargs...)
    p3 = plot(itp_linear; title="Linear Interpolation\n (Error≈$(measure_error_string(itp_linear)))", kwargs..., itp_plot_kwargs...)
    p4 = plot(itp_cubic; title="Cubic Interpolation\n (Error≈$(measure_error_string(itp_cubic)))", kwargs..., itp_plot_kwargs...)

    return plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 800), dpi=200)
end


x_test = [0, 0.1, 0.4, 0.5, 0.82, 1.0]
y_test = [0, 0.1, 0.2,0.5,  0.8, 0.9,1.0]
p = plot_comparison(x_test, y_test)

savefig(p, "$( pkgdir(FastInterpolations))/docs/images/readme_2d_comparison.png")



# function chebyshev_nodes(n, a=0.0, b=1.0)
#     nodes = [(a + b)/2 + (b - a)/2 * cos((2k - 1) * π / (2n)) for k in n:-1:1]
#     sort!(nodes)
#     nodes[1] = a
#     nodes[end] = b
#     return nodes
# end

# nx, ny = 5, 5

# # High-resolution evaluation
# x_hi = range(0.0,1.0,length=300) 
# y_hi = range(0.0,1.0,length=300) 

# x_uni= range(0.0, 1.0, length=nx)
# y_uni = range(0.0, 1.0, length=ny)

# x_ch = chebyshev_nodes(nx, 0.0, 1.0)
# y_ch = chebyshev_nodes(ny, 0.0, 1.0)

# xs, ys = x_uni, y_uni

