# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    DEPRECATED IMPLEMENTATIONS                              ║
# ║           (Cleaned up after SeriesInterpolant migration)                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# The old composition-based MultiInterpolant implementations have been removed.
# They were replaced by unified-matrix SeriesInterpolant which provides:
# - 10-120x faster scalar evaluation (SIMD point-contiguous access)
# - Competitive vector evaluation (anchor pooling)
# - Simpler, single-struct storage
#
# Backward compatibility is maintained via type aliases:
#   LinearMultiInterpolant = LinearSeriesInterpolant
#   ConstantMultiInterpolant = ConstantSeriesInterpolant
#   QuadraticMultiInterpolant = QuadraticSeriesInterpolant
#   CubicMultiInterpolant = CubicSeriesInterpolant
#
