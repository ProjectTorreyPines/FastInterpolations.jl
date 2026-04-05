module FastInterpolations

import LinearAlgebra
import Printf
using LinearAlgebra: Tridiagonal, lu, ldiv!
using Preferences: @load_preference, @set_preferences!
using AdaptiveArrayPools
using HelpPlots: help_plot, help_plot!

# Module aggregators (encapsulate include order complexity)
include("core/core.jl")
include("linear/linear.jl")
include("constant/constant.jl")
include("quadratic/quadratic.jl")
include("cubic/cubic.jl")
include("hermite/hermite.jl")
include("pchip/pchip.jl")
include("cardinal/cardinal.jl")
include("akima/akima.jl")
include("hetero/hetero.jl")

# Derivative view wrapper (depends on all interpolant types)
include("derivative_view.jl")

# Vector calculus operations (depends on ND interpolant types)
include("vector_calculus.jl")

# Method traits: constructor/adjoint dispatch mappings (shared by AD extensions)
include("method_traits.jl")

# Custom show methods (depends on all interpolant types and DerivativeView)
include("core/show.jl")

# Integration API (depends on all interpolant types)
include("integral/integral.jl")

# Polynomial coefficient extraction (depends on all interpolant types)
include("coeffs.jl")

# Nodal partial derivative access (zero-copy views into precomputed partials)
include("nodal_partials.jl")

# Exports
export AbstractInterpolant, AbstractInterpolant1D, AbstractHermiteInterpolant1D, AbstractSeriesInterpolant, AbstractInterpolantND, AbstractAdjoint, AbstractAdjoint1D, AbstractAdjointND
export Series, n_series
export grid_type, value_type, eval_type  # Type introspection for {Tg, Tv} system
export linear_interp, linear_interp!, LinearInterpolant, LinearSeriesInterpolant, LinearInterpolantND
export constant_interp, constant_interp!, ConstantInterpolant, ConstantSeriesInterpolant, ConstantInterpolantND
export quadratic_interp, quadratic_interp!, QuadraticInterpolant, QuadraticSeriesInterpolant, QuadraticInterpolantND
export cubic_interp, cubic_interp!, CubicSplineCache, CubicInterpolant, CubicSeriesInterpolant
export hermite_interp, hermite_interp!, CubicHermiteInterpolant1D  # Cubic Hermite with user-supplied slopes
export pchip_interp, pchip_interp!, PchipInterpolant1D  # PCHIP monotone interpolation
export cardinal_interp, cardinal_interp!, CardinalInterpolant1D  # Cardinal spline (CatmullRom default)
export akima_interp, akima_interp!, AkimaInterpolant1D  # Akima outlier-robust interpolation
export constant_adjoint, ConstantAdjoint, ConstantAdjointND  # Constant adjoint operators (W^T)
export linear_adjoint, LinearAdjoint, LinearAdjointND  # Linear adjoint operators (W^T)
export quadratic_adjoint, QuadraticAdjoint, QuadraticAdjointND  # Quadratic adjoint operators (W^T)
export cubic_adjoint, CubicAdjoint, CubicAdjointND  # Cubic adjoint operators (W^T)
export hetero_adjoint, HeteroAdjointND  # Heterogeneous ND adjoint operator (W^T)
export hermite_adjoint, HermiteAdjoint1D  # Hermite adjoint (user-supplied slopes)
export pchip_adjoint, PchipAdjoint1D  # PCHIP adjoint (slope-from-data, data-dependent)
export cardinal_adjoint, CardinalAdjoint1D  # Cardinal adjoint (slope-from-data)
export akima_adjoint, AkimaAdjoint1D  # Akima adjoint (slope-from-data, data-dependent)
export CubicInterpolantND, AbstractCoeffStrategy, PreCompute, OnTheFly  # ND cubic types
export interp, interp!, HeteroInterpolantND  # Tensor product ND (per-axis methods)
export AbstractInterpMethod, CubicInterp, LinearInterp, QuadraticInterp, ConstantInterp, NoInterp
export GridIdx
export gradient, gradient!, value_gradient, hessian, hessian!, laplacian  # Analytical vector calculus for ND
export precompute_transpose!  # Pre-allocate point-contiguous layout for scalar queries
export set_cubic_cache_size!, get_cubic_cache_size, clear_cubic_cache!

# Search policy types (user-facing algorithm selection)
export AbstractSearchPolicy, BinarySearch, LinearSearch, LinearBinarySearch, AutoSearch

# Boundary condition types
export AbstractBC, PointBC, Deriv1, Deriv2, Deriv3, BCPair
export ZeroCurvBC, ZeroSlopeBC, PeriodicBC, MinCurvFit
export PolyFit, LinearFit, QuadraticFit, CubicFit  # Polynomial fitting BCs
export Left, Right

# Derivative view functions and types
export deriv1, deriv2, deriv3, deriv_view, AbstractDerivativeView, DerivativeView

# Operation types (for derivative dispatch)
export AbstractEvalOp, DerivOp, deriv_order
export EvalValue, EvalDeriv1, EvalDeriv2, EvalDeriv3  # backward-compat aliases

# Extrapolation mode types (typed API for ND extrap)
export AbstractExtrap, NoExtrap, ConstExtrap, ClampExtrap, FillExtrap, ExtendExtrap, WrapExtrap, InBounds

# Side selection types (constant interpolation)
export AbstractSide, NearestSide, LeftSide, RightSide

# Factory functions (high-level convenience API)
export Search, Extrap, Side

# Integration
export integrate, cumulative_integrate

# Polynomial coefficient extraction
export CellPoly, coeffs
export nodal_partials

# Plot recipe helpers (re-exported from HelpPlots)
export help_plot, help_plot!

end # module
