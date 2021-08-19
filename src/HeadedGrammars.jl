module Headed

using AbstractGrammars
import AbstractGrammars: apply, initial_category, push_completions!
using Test

# check for the tag of an object
⊣(tag, x) = x.tag == tag

default(::Type{T}) where T <: Number = zero(T)
default(::Type{Symbol}) = Symbol()
# default(::Type{T}) where T <: AbstractString = one(T)
# default(::Type{T}) where T <: AbstractVector = T()

@enum CategoryTag start terminal nonterminal

struct Category{T,NT}
  tag     :: CategoryTag
  tlabel  :: T
  ntlabel :: NT
end

start_category(T, NT) = Category(start, default(T), default(NT))
terminal_category(NT, t) = Category(terminal, t, default(NT))
nonterminal_category(T, nt) = Category(nonterminal, default(T), nt)

default(::Type{Category{T,NT}}) where {T,NT} = start_category(T, NT)



@enum RuleTag startrule terminate duplicate leftheaded rightheaded 

struct Rule{T,NT} <: AbstractRule{Category{T,NT}}
  tag      :: RuleTag
  category :: Category{T,NT}
end

start_rule(c) = Rule(startrule, c)
termination_rule(T, NT) = Rule(terminate, default(Category{T, NT}))
duplication_rule(T, NT) = Rule(duplicate, default(Category{T, NT}))
leftheaded_rule(c) = Rule(leftheaded, c)
rightheaded_rule(c) = Rule(rightheaded, c)

# tests
T, NT = Float64, Int
@test isbitstype(Category{T,NT})
@test isbitstype(Rule{T,NT})

struct Grammar{T,NT,TT,FT} <: AbstractGrammar{Rule{T,NT}}
  rules        :: Set{Rule{T,NT}}
  toTerminal   :: TT # function NonTerminal{T,NT} -> Terminal{T,NT}
  fromTerminal :: FT # function Terminal{T,NT} -> Vector{NonTerminal{T,NT}}
end

function apply(grammar::Grammar, r::Rule, c::Category)
  duplicate   == r.tag                    && return (c, c)
  leftheaded  == r.tag && c != r.category && return (c, r.category)
  rightheaded == r.tag && c != r.category && return (r.category, c)
  start_rule  == r.tag && start == c.tag  && return (r.category,)
  terminate   == r.tag                    && return (grammar.toTerminal(c),)
  return nothing
end

initial_category(::Grammar{T,NT,TT,FT}) where {T,NT,TT,FT} = 
  start_category(T, NT)
duplication_rule(::Grammar{T,NT,TT,FT}) where {T,NT,TT,FT} = 
  duplication_rule(T, NT)
termination_rule(::Grammar{T,NT,TT,FT}) where {T,NT,TT,FT} =
  termination_rule(T, NT)
 
function push_completions!(grammar::Grammar, stack, c)
  if terminal ⊣ c
    for lhs in grammar.fromTerminal(c)
      push!(stack, App(grammar, lhs, termination_rule(grammar)))
    end
  elseif nonterminal ⊣ c && start_rule(c) in grammar.rules
    push!(stack, App(grammar, initial_category(grammar), start_rule(c)))
  end
end

function push_completions!(grammar::Grammar, stack, c1, c2)
  nonterminal ⊣ c1 && nonterminal ⊣ c2 || return nothing
  if c1 == c2
    r = duplication_rule(grammar)
    if r in grammar.rules
      push!(stack, App(grammar, c1, r))
    end
  else
    r = leftheaded_rule(c2)
    if r in grammar.rules
      push!(stack, App(grammar, c1, r))
    end
    r = rightheaded_rule(c1)
    if r in grammar.rules
      push!(stack, App(grammar, c2, r))
    end
  end
end

# tests
T, NT = String, Symbol
nt = nonterminal_category(T, :bar)
lhr = leftheaded_rule(nonterminal_category(T, :foo))
grammar = Grammar(Set([lhr]), nothing, nothing)
@test apply(grammar, lhr, nt) == nonterminal_category.(T, (:bar, :foo))



T, NT = Int, Int
nonterminals = nonterminal_category.(T, 1:4)
terminals = terminal_category.(NT, 1:3)
termination_dict = Dict(1 => 1, 2 => 2, 3 => 3, 4 => 3)
toTerminal(c) = terminal_category(NT, termination_dict[c.ntlabel])
inv_termination_dict = Dict(1 => [1], 2 => [2], 3 => [3, 4])
fromTerminal(c) = nonterminal_category.(T, inv_termination_dict[c.tlabel])
rules = Set{Rule{T,NT}}([
  start_rule(first(nonterminals));
  termination_rule(T, NT);
  duplication_rule(T, NT); 
  leftheaded_rule.(nonterminals); 
  rightheaded_rule.(nonterminals)])
grammar = Grammar(rules, toTerminal, fromTerminal)

import Distributions: logpdf
logpdf(::Grammar, lhs, rule) = log(0.5)

terminalss = [[rand(terminals)] for _ in 1:70]
@time chart = chartparse(grammar, InsideScoring(), terminalss)
chart[1,70][initial_category(grammar)]



T, NT = Float64, Int
nonterminals = nonterminal_category.(T, 1:100)
terminals = terminal_category.(NT, 1:100)
toTerminal(c) = terminal_category(NT, float(c.ntlabel))
fromTerminal(c) = [nonterminal_category(T, Int(c.tlabel))]
rules = Set{Rule{T,NT}}([
  start_rule.(nonterminals);
  termination_rule(T, NT);
  duplication_rule(T, NT); 
  # leftheaded_rule.(nonterminals); 
  rightheaded_rule.(nonterminals)])
grammar = Grammar(rules, toTerminal, fromTerminal)


import ProfileVega

terminalss = [[rand(terminals)] for _ in 1:100]
@time chart = chartparse(grammar, CountScoring(), terminalss)
chart[1,100][initial_category(grammar)]
scoring = AbstractGrammars.WDS(grammar)
@time chart = chartparse(grammar, scoring, terminalss)
chart[1,40][initial_category(grammar)]
s
scoring.store
eltype(scoring.store) |> isbitstype

@time chart = chartparse(grammar, AbstractGrammars.TreeDistScoring(), terminalss)



# function testfill!(xs, labels)
#   for i in eachindex(xs)
#     xs[i] = (labels[rand(1:length(labels))], rand(Int))
#   end
# end

# labels = [:foo, :bar, :baz]
# xs = Vector{Tuple{Symbol, Int}}(undef, 1000)
# @time testfill!(xs, labels)

# labels = ["foo", "bar", "baz"]
# xs = Vector{Tuple{String, Int}}(undef, 1000)
# @time testfill!(xs, labels)

# labels = ["foo", :bar, 42.0]
# xs = Vector{Tuple{Any, Int}}(undef, 1000)
# @time testfill!(xs, labels)

# labels = Union{String, Symbol}["foo", :bar, :baz]
# xs = Vector{Tuple{Union{String, Symbol}, Int}}(undef, 1000)
# @time testfill!(xs, labels)

# labels = Union{Int, Bool}[42, true, false]
# xs = Vector{Tuple{Union{Int, Bool}, Int}}(undef, 1000)
# @time testfill!(xs, labels)

# labels = Union{Nothing, Bool}[nothing, true, false]
# xs = Vector{Tuple{Union{Nothing, Bool}, Int}}(undef, 1000)
# @time testfill!(xs, labels)



end # module
