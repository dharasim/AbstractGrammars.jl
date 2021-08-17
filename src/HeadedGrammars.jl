module Headed

using AbstractGrammars
import AbstractGrammars: apply, initial_category, push_completions!
using Test

# const tests = []

struct Start{T,NT}                 end
struct Terminal{T,NT}    label::T  end
struct NonTerminal{T,NT} label::NT end
const Category{T,NT} = Union{Start{T,NT}, Terminal{T,NT}, NonTerminal{T,NT}}

abstract type URule{T,NT} <: AbstractRule{Category{T,NT}} end
struct Termination{T,NT}  <: URule{T,NT} end
struct StartRule{T,NT}    <: URule{T,NT} category::NonTerminal{T,NT} end

abstract type BRule{T,NT} <: AbstractRule{Category{T,NT}} end
struct LeftHeaded{T,NT}   <: BRule{T,NT} dependent::NonTerminal{T,NT} end
struct RightHeaded{T,NT}  <: BRule{T,NT} dependent::NonTerminal{T,NT} end
struct Duplication{T,NT}  <: BRule{T,NT} end

const Rule{T,NT} = Union{
  Termination{T,NT}, StartRule{T,NT}, 
  LeftHeaded{T,NT}, RightHeaded{T,NT}, Duplication{T,NT}}

T, NT = Float64, Int
@test isplain(Category{T,NT})
@test isplain(Rule{T,NT})

struct Grammar{T,NT,TT,FT} <: AbstractGrammar{Rule{T,NT}}
  rules        :: Set{Rule{T,NT}}
  toTerminal   :: TT # function NonTerminal{T,NT} -> Terminal{T,NT}
  fromTerminal :: FT # function Terminal{T,NT} -> Vector{NonTerminal{T,NT}}
end

# by default a rule is not applicable
apply(::Rule, ::Category) = nothing
apply(r::LeftHeaded, c::NonTerminal) =
  c == r.dependent ? nothing : (c, r.dependent)
apply(r::RightHeaded, c::NonTerminal) = 
  c == r.dependent ? nothing : (r.dependent, c)
apply(::Duplication, c::NonTerminal) = (c, c)
apply(r::StartRule, ::Start) = (r.category,)
apply(g::Grammar, ::Termination, c::NonTerminal) = (g.toTerminal(c),)

initial_category(::Grammar{T,NT,TT,FT}) where {T,NT,TT,FT} = Start{T,NT}()
duplication(::Grammar{T,NT,TT,FT}) where {T,NT,TT,FT} = Duplication{T,NT}()
termination(::Grammar{T,NT,TT,FT}) where {T,NT,TT,FT} = Termination{T,NT}()

function push_completions!(grammar::Grammar, stack, c::Terminal)
  for lhs in grammar.fromTerminal(c)
    push!(stack, App(grammar, lhs, termination(grammar)))
  end
end

function push_completions!(grammar::Grammar, stack, c::NonTerminal)
  if StartRule(c) in grammar.rules
    push!(stack, App(grammar, initial_category(grammar), StartRule(c)))
  end
end

function push_completions!(
  grammar::Grammar, stack, c1::NonTerminal, c2::NonTerminal
)
  if c1 == c2
    if duplication(grammar) in grammar.rules
      push!(stack, App(grammar, c1, duplication(grammar)))
    end
  else
    if LeftHeaded(c2) in grammar.rules
      push!(stack, App(grammar, c1, LeftHeaded(c2)))
    end
    if RightHeaded(c1) in grammar.rules
      push!(stack, App(grammar, c2, RightHeaded(c1)))
    end
  end
end

function push_completions!(grammar::Grammar, stack, c1, c2)
  nothing
end

begin
  T, NT = String, Symbol
  nt = NonTerminal{T,NT}(:bar)
  LeftHeaded(nt)
  lhr = LeftHeaded{T,NT}(NonTerminal{T,NT}(:foo))
  @test apply(lhr, nt) == map(NonTerminal{T,NT}, (:bar, :foo))
end






T, NT = Int, Int
nonterminals = map(NonTerminal{T, NT}, 1:4)
terminals = map(Terminal{T, NT}, 1:3)
termination_dict = Dict(1 => 1, 2 => 2, 3 => 3, 4 => 3)
toTerminal(c) = Terminal{T,NT}(termination_dict[c.label])
inv_termination_dict = Dict(1 => [1], 2 => [2], 3 => [3, 4])
fromTerminal(c) = map(NonTerminal{T, NT}, inv_termination_dict[c.label])
rules = Set{Rule{T,NT}}([
  StartRule(first(nonterminals));
  Termination{T,NT}();
  Duplication{T,NT}(); 
  map(LeftHeaded, nonterminals); 
  map(RightHeaded, nonterminals)])
grammar = Grammar(rules, toTerminal, fromTerminal)

import Distributions: logpdf
logpdf(::Grammar, lhs, rule) = log(0.5)

terminalss = [[rand(terminals)] for _ in 1:70]
@time chart = chartparse(grammar, InsideScoring(), terminalss)
chart[1,70][initial_category(grammar)]



T, NT = Float64, Int
nonterminals = map(NonTerminal{T, NT}, 1:100)
terminals = map(Terminal{T, NT}, 1:100)
toTerminal(c) = Terminal{T,NT}(float(c.label))
fromTerminal(c) = [NonTerminal{T, NT}(Int(c.label))]
rules = Set{Rule{T,NT}}([
  StartRule(first(nonterminals));
  Termination{T,NT}();
  Duplication{T,NT}(); 
  map(LeftHeaded, nonterminals); 
  map(RightHeaded, nonterminals)])
grammar = Grammar(rules, toTerminal, fromTerminal)


import ProfileVega

terminalss = [[rand(terminals)] for _ in 1:40]
@time chart = chartparse(grammar, CountScoring(), terminalss)
chart[1,50][initial_category(grammar)]
scoring = AbstractGrammars.TDS(grammar)
@ProfileVega.profview chart = chartparse(grammar, scoring, terminalss)
idx = chart[1,50][initial_category(grammar)]
scoring.store
eltype(scoring.store) |> isplain

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
