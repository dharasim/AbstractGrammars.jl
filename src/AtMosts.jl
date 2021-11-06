#######################################
### Short lists of limited max size ###
#######################################

module AtMosts

using ..AbstractGrammars: default

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

# function ==(xs::T1, ys::T2) where {T1 <: AtMost, T2 <: AtMost}
#   xs.length == ys.length && all(isequal(x, y) for (x, y) in zip(xs, ys))
# end

# function hash(xs::AtMost, h::UInt)
#   result = h
#   for x in xs
#     result = hash(x, result)
#   end
#   return result
# end

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


# function generate_code_atmost(n)
#   struct_expr = quote
#     struct $(Symbol(:AtMost, n)){T} <: AtMost{T}
#       length::Length
#     end
#   end
#   append!( # append field expressions
#     struct_expr.args[2].args[3].args, 
#     [:($(Symbol(:val, i)) :: T) for i in 1:n])
  
#   struct_args = struct_expr.args[2].args[3].args
#   ex = quote
#     function $(Symbol(:AtMost, n))(T::Type)
#       new{T}(Length(0))
#     end
#   end
#   push!(struct_args, ex.args[2])
#   for k in 1:n
#     vals = Symbol.(:val, 1:k)
#     typed_vals = Expr.(:(::), vals, :T)
#     ex = quote
#       function $(Symbol(:AtMost, n))($(typed_vals...),) where T
#         new{T}(Length($k), $(vals...))
#       end
#     end
#     push!(struct_args, ex.args[2])
#   end

#   # ex = Expr(:block, struct_expr, constr_exprs)
#   eval(struct_expr)
# end

# for n in 0:10
#   generate_code_atmost(n)
# end

end # module