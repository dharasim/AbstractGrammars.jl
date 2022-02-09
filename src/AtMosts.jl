#######################################
### Short lists of limited max size ###
#######################################

module AtMosts

using ..AbstractGrammars: default
# default(::Type{Char}) = 'a'

import Base: length, getindex, iterate, eltype, show, keys

export AtMost, atmost1, atmost2, atmost3, atmost4, atmost5, atmost6, atmost7, atmost8, atmost9, atmost10

struct Length val::Int end

struct AtMost{T, N}
  length :: Length
  vals   :: NTuple{N, T}

  function AtMost(xs::T...; limit::Int) where T
    k = length(xs)
    @assert k <= limit
    new{T, limit}(Length(k), tuple(xs..., ntuple(i -> default(T), limit-k)...))
  end
end

length(xs::AtMost) = xs.length.val
getindex(xs::AtMost, i) = getfield(xs.vals, i)
keys(xs::AtMost) = 1:length(xs)
eltype(::Type{A}) where {T, A <: AtMost{T}} = T

function iterate(xs::AtMost, i=0)
  if i == length(xs)
    nothing
  else
    (xs[i+1], i+1)
  end
end

function show(io::IO, xs::AtMost{T, N}) where {T, N}
  print(io, "AtMost{$T, $N}(")
  l = length(xs)
  for i in 1:l-1
    print(io, xs[i], ",")
  end
  print(io, xs[l], ")")
end

for N in 1:10
  @eval $(Symbol(:atmost, N))(xs...) = AtMost(xs..., limit=$N)
end

end # module