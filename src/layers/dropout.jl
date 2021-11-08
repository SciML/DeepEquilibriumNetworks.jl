
# # _isactive(m) = isnothing(m.active) ? istraining() : m.active

# # _dropout_shape(s, ::Colon) = size(s)
# # _dropout_shape(s, dims) = tuple((i ∉ dims ? 1 : si for (i, si) ∈ enumerate(size(s)))...)

# # _dropout_kernel(y::T, p, q) where {T} = y > p ? T(1 / q) : T(0)

# mutable struct VariationalHiddenDropout{F,D,M}
#     p::F
#     dims::D
#     mask::M
#     active::Union{Bool,Nothing}
# end

# function VariationalHiddenDropout(p, s; dims = :)
#     @assert 0 ≤ p ≤ 1
#     mask = zeros
#     return VariationalHiddenDropout(p, dims, nothing)
# end

# function (a::VariationalHiddenDropout)(x)
#     Flux._isactive(a) || return x
#     return variational_hidden_dropout(x, a.p; dims = a.dims, active = true)
# end

# Flux.testmode!(m::VariationalHiddenDropout, mode = true) =
#     (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

# function Base.show(io::IO, d::VariationalHiddenDropout)
#     print(io, "VariationalDropout(", d.p)
#     d.dims != (:) && print(io, ", dims = $(repr(d.dims))")
#     print(io, ")")
# end
