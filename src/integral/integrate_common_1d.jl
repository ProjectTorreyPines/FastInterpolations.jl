@inline function _normalize_bounds_1d(a::Real, b::Real)
    if a < b
        return (1, a, b)
    elseif a > b
        return (-1, b, a)
    else
        return (0, a, b)
    end
end
