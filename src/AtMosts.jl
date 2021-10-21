#######################################
### Short lists of limited max size ###
#######################################

module AtMosts

import Base: length, getindex, iterate, eltype, show

export AtMost, AtMost0, AtMost1, AtMost2, AtMost3, AtMost4

abstract type AtMost{T} end

length(xs::AtMost) = xs.length.val
getindex(xs::AtMost, i) = getfield(xs, i+1)
eltype(::Type{A}) where {T, A <: AtMost{T}} = T

function iterate(xs::AtMost, i=0)
  if i == length(xs)
    nothing
  else
    (xs[i+1], i+1)
  end
end

function show(io::IO, iter::AtMost{T}) where T
  print(io, "AtMost{$T}(")
  for x in iter
    print(io, x, ",")
  end
  print(io, ")")
end

struct Length val::Int end

function generate_code_atmost(n)
  struct_expr = quote
    struct $(Symbol(:AtMost, n)){T} <: AtMost{T}
      length::Length
    end
  end
  append!( # append field expressions
    struct_expr.args[2].args[3].args, 
    [:($(Symbol(:val, i)) :: T) for i in 1:n])
  
  struct_args = struct_expr.args[2].args[3].args
  ex = quote
    function $(Symbol(:AtMost, n))(T::Type)
      new{T}(Length(0))
    end
  end
  push!(struct_args, ex.args[2])
  for k in 1:n
    vals = Symbol.(:val, 1:k)
    typed_vals = Expr.(:(::), vals, :T)
    ex = quote
      function $(Symbol(:AtMost, n))($(typed_vals...),) where T
        new{T}(Length($k), $(vals...))
      end
    end
    push!(struct_args, ex.args[2])
  end

  # ex = Expr(:block, struct_expr, constr_exprs)
  eval(struct_expr)
end

for n in 0:4
  generate_code_atmost(n)
end

end # module