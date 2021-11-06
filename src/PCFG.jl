module PCFG

using AbstractGrammars
using AbstractGrammars.AtMosts
using Test

import Base: show
import AbstractGrammars: apply, arity

export Rule, -->, treelet2cfrule

struct Rule{C} <: AbstractRule{C}
  lhs :: C
  rhs :: AtMost{C, 2}
end

Rule(lhs, rhs...) = Rule(lhs, atmost2(rhs...))
-->(lhs::C, rhs::C) where C = Rule(lhs, rhs)
-->(lhs::C, rhs::Union{Tuple{C}, Tuple{C, C}}) where C = Rule(lhs, rhs...)

apply(r::Rule{C}, c::C) where C = r.lhs == c ? tuple(r.rhs...) : nothing
arity(r::Rule) = length(r.rhs)

function show(io::IO, r::Rule{C}) where C
  print(io, "Rule{$C}($(r.lhs) -->")
  foreach(c -> print(io, " $c"), r.rhs)
  print(io, ")")
end

function treelet2cfrule(treelet::Treelet)
  @assert arity(treelet) in (1, 2)
  Rule(treelet.root_label, treelet.child_labels...)
end

end # module