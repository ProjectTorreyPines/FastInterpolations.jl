"""
    FastInterpolations.Experimental

Features whose API and behaviour may change, or be removed, in any release. Names
here are not covered by semantic versioning; bring them in explicitly with
`using FastInterpolations.Experimental`.

Currently provided:
- `phs_interp`, `phs_interp!`, `PHSInterpolantND` — polyharmonic spline interpolation
  on uniform rectilinear grids.
"""
module Experimental

using ..FastInterpolations: phs_interp, phs_interp!, PHSInterpolantND

export phs_interp, phs_interp!, PHSInterpolantND

end # module Experimental
