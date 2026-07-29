"""
Hessian Evaluation Performance Analysis
=========================================

Based on CPU profiling results from profile_bottlenecks.jl:

CURRENT APPROACH (Three Separate Calls):
  • 201 samples in _phs_eval_blended for Laplacian evaluation
  • Each call iterates through 27 blend nodes independently
  • Per-node operations (weight calc, stencil solve) repeated 3×
  • Execution time: ~5.1 ms per 1000 query points (5.1 μs per point)

PROPOSED FUSED APPROACH (Single Call):
  • Single iteration through 27 blend nodes
  • Pre-compute weights/derivatives once, reuse for all 3 components
  • Expected reduction: ~67 samples (3× improvement)
  • Expected speedup: 2.5-3× (5.1 → 1.7-2.0 ms per 1000 points)

KEY INSIGHT:
  The _phs_eval_blended_G_with_hess function ALREADY EXISTS in phs_eval.jl
  but is only called internally for single-point evaluation via @with_pool.
  
  We need to:
  1. Expose a batch version for array evaluation
  2. Update dispatch to use it when deriv=(D2, D2, D2) is requested

EXPECTED OUTCOME:
  • Laplacian evaluation: 5.1 ms → ~1.8 ms (2.8× speedup)
  • Batch density comparison run: ~7 seconds saved per evaluation
  • No changes to API or correctness
"""

println(__doc__)
