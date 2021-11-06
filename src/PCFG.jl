module PCFG

end # module

#############
### Trees ###
#############

include("AbstractGrammars.jl")
using .AbstractGrammars

import Base: map, eltype, show
import AbstractGrammars: apply

struct Treee{T}
  label :: T
  children :: Vector{Treee{T}}
end

Treee(label::T) where T = Treee(label, Treee{T}[])
isleaf(tree::Treee) = isempty(tree.children)
map(f, tree::Treee) = Treee(f(tree.label), map(f, tree.children))
eltype(::Type{Treee{T}}) where T = T

function dict2tree(f, dict; label_key="label", children_key="children")
  label = f(dict[label_key])
  children = [dict2tree(f, child) for child in dict[children_key]]
  Treee(label, children)
end

function apply(derivation::Vector{<:AbstractRule}, c)
  i = 0 # rule index
  next_rule() = (i += 1; derivation[i])
  backtrack() = (i -= 1)
  are_there_more_rules() = (i < length(derivation))

  function rewrite(lhs)
    r = next_rule()
    rhs = apply(r, lhs)
    if isnothing(rhs) # rule r is not applicable to lhs
      backtrack()
      return Treee(lhs)
    else # rule r is applicable to lhs
      children = [are_there_more_rules() ? rewrite(c) : Treee(c) for c in rhs]
      return Treee(lhs, children)
    end
  end

  rewrite(c)
end

struct Treelet{T}
  root_label   :: T
  child_labels :: Vector{T}
end

function Treelet(root_label::T, child_labels::T...) where T
  Treelet(root_label, collect(child_labels))
end

arity(treelet::Treelet) = length(treelet.child_labels)

function treelets(tree::Treee{T}, out=Treelet{T}[]) where T
  treelet = Treelet(tree.label, map(child -> child.label, tree.children))
  push!(out, treelet)
  foreach(child -> treelets(child, out), tree.children)
  return out
end

labels(tree::Treee) = 
  [treelet.root_label for treelet in treelets(tree)]
leaflabels(tree::Treee) = 
  [treelet.root_label for treelet in treelets(tree) if arity(treelet) == 0]
innerlabels(tree::Treee) = 
  [treelet.root_label for treelet in treelets(tree) if arity(treelet) > 0]

tree2derivation(treelet2rule, tree::Treee) = 
  [treelet2rule(tl) for tl in treelets(tree) if arity(tl) >= 1]

#############
### Rules ###
#############

using AbstractGrammars.AtMosts
using Test

export Rule, -->

struct Rule{C} <: AbstractRule{C}
  lhs :: C
  rhs :: AtMost2{C}
end

Rule(lhs, rhs...) = Rule(lhs, AtMost2(rhs...))
-->(lhs::C, rhs::C) where C = Rule(lhs, rhs)
-->(lhs::C, rhs::Union{Tuple{C}, Tuple{C, C}}) where C = Rule(lhs, rhs...)

apply(r::Rule{C}, c::C) where C = r.lhs == c ? tuple(r.rhs...) : nothing
arity(r::Rule) = length(r.rhs)

function show(io::IO, r::Rule{C}) where C
  print(io, "Rule{$C}($(r.lhs) -->")
  foreach(c -> print(io, " $c"), r.rhs)
  print(io, ")")
end



function treelet2rule(treelet::Treelet)
  @assert arity(treelet) in (1, 2)
  Rule(treelet.root_label, treelet.child_labels...)
end




### @testset "parse trees"

derivation = [
    'A' --> 'B', 
    'B' --> ('C', 'D'),
    'C' --> 'c',
    'D' --> ('B', 'A'),
    'B' --> 'b',
    'A' --> 'a']

tree = apply(derivation, 'A')
@test tree isa Treee{Char}
@test Char == eltype(tree)
@test labels(tree) == ['A', 'B', 'C', 'c', 'D', 'B', 'b', 'A', 'a']
@test leaflabels(tree) == ['c', 'b', 'a']
@test innerlabels(tree) == ['A', 'B', 'C', 'D', 'B', 'A']

derivation2 = tree2derivation(treelet2rule, tree)

default

derivation[1] === derivation2[1]
isbits(derivation[1].rhs)
xs = derivation[1] 
ys = derivation2[1]
xs.length == ys.length && all(isequal(x, y) for (x, y) in zip(xs, ys))

struct Fooo
  val :: Vector{Int}
  Fooo(val) = new(val)
  Fooo() = new()
end




derivation[1].rhs == derivation2[1].rhs
derivation[1] == derivation2[1]

hash(derivation2[1].rhs), hash(derivation[1].rhs)
derivation[1].rhs.val2
[1,2] == [1.0, 2.0]



@testset "standard context-free rules" begin
  r = 'a' --> ('b', 'c')
  @test isbits(r) # ensure that r is stack-allocated
  @test apply(r, 'a') == ('b', 'c')
  @test apply(r, 'b') === nothing
  @test arity(r) == 2

  derivation = [
    'A' --> 'B', 
    'B' --> ('C', 'D'),
    'C' --> 'c',
    'D' --> ('B', 'A'),
    'B' --> 'b',
    'A' --> 'a']
  @test arity.(derivation) == [1, 2, 1, 2, 1, 1]

  tree = apply(derivation, 'A')
  @test leaflabels(tree) == ['c', 'b', 'a']
end





### Re-implementation AtMost

using AbstractGrammars

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

AtMost(1, 2, limit=5)

for N in 1:10
  @eval $(Symbol(:atmost, N))(xs...) = AtMost(xs..., limit=$N)
end

atmost3(4, 5)


atmost4()



AtMost(1, 2, limit=3)

AtMost{3}
AtMost{3}(1,2)

AtMost{3}(Length(2), (3,4))
