"""
    FastInterpolations.Experimental

Features whose API and behaviour may change, or be removed, in any release. Names
here are not covered by semantic versioning; bring them in explicitly with
`using FastInterpolations.Experimental`.

Names exported here:
- `phs_interp`, `phs_interp!`, `PHSInterpolantND` — polyharmonic spline interpolation
  on uniform rectilinear grids.

Experimental features reached through existing names (nothing to import from here):
- GPU arrays: the existing API on device arrays, active once a GPU package is loaded
  (Adapt / GPUArrays extensions). See the GPU Arrays guide for what may change.
"""
module Experimental

using ..FastInterpolations: phs_interp, phs_interp!, PHSInterpolantND

export phs_interp, phs_interp!, PHSInterpolantND

end # module Experimental
