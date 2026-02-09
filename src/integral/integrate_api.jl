function integrate(itp::AbstractInterpolant)
    throw(ArgumentError("integrate(itp) is not implemented for $(typeof(itp)) yet"))
end

function integrate(itp::AbstractInterpolant, x0::Real, x1::Real; search=nothing, hint=nothing)
    throw(ArgumentError("integrate(itp, x0, x1) is not implemented for $(typeof(itp)) yet"))
end

function integrate(itp::AbstractInterpolantND{Tg,Tv,N}) where {Tg,Tv,N}
    throw(ArgumentError("integrate(itp_nd) is not implemented for $(typeof(itp)) yet"))
end

function integrate(
    itp::AbstractInterpolantND{Tg,Tv,N},
    lo::NTuple{N,<:Real},
    hi::NTuple{N,<:Real};
    search=nothing,
    hint=nothing
) where {Tg,Tv,N}
    throw(ArgumentError("integrate(itp_nd, lo, hi) is not implemented for $(typeof(itp)) yet"))
end
