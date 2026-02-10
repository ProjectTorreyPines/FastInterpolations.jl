# Private internal op tags for integrate kernels — not exported
abstract type _AbstractIntegralOp end
struct _EvalIntegralPartial <: _AbstractIntegralOp end
struct _EvalIntegralCell <: _AbstractIntegralOp end
