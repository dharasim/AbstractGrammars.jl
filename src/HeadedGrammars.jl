module Headed

using AbstractGrammars
import AbstractGrammars: initial_category, push_completions!

abstract type Category{T,NT} end
struct Start{T,NT}         <: Category{T,NT} end
struct Terminal{T,NT}      <: Category{T,NT} label::T end
struct NonTerminal{T,NT}   <: Category{T,NT} label::NT end

abstract type Rule{T,NT}    <: AbstractRule{Category{T,NT}} end
struct Termination{T,NT,TT} <: Rule{T,NT} toTerminal::TT end
struct StartRule{T,NT}      <: Rule{T,NT} category::NonTerminal{T,NT} end

abstract type NTRule{T,NT} <: Rule{T,NT} end
struct LeftHeaded{T,NT}    <: NTRule{T,NT} dependent::NonTerminal{T,NT} end
struct RightHeaded{T,NT}   <: NTRule{T,NT} dependent::NonTerminal{T,NT} end
struct Duplication{T,NT}   <: NTRule{T,NT} end

# by default a rule is not applicable
(r::Rule)(c::Category) = nothing

(r::LeftHeaded)(c::NonTerminal)  = c == r.dependent ? nothing : (c, r.dependent)
(r::RightHeaded)(c::NonTerminal) = c == r.dependent ? nothing : (r.dependent, c)
(::Duplication)(c::NonTerminal)  = (c, c)
(r::StartRule)(::Start)          = (r.category,)
(r::Termination)(c::NonTerminal) = r.toTerminal(c)

struct Grammar{T,NT,TT,FT} <: AbstractGrammar{Rule{T,NT}}
  nt_rules         :: Set{NTRule{T,NT}}
  start_categories :: Set{NonTerminal{T,NT}}
  termination      :: Termination{T,NT,TT}
  fromTerminal     :: FT
end

initial_category(::Grammar{T,NT,TT,FT}) where {T,NT,TT,FT} = Start{T,NT}()
duplication(::Grammar{T,NT,TT,FT}) where {T,NT,TT,FT} = Duplication{T,NT}()

function push_completions!(grammar::Grammar, stack, c::Terminal)
  for lhs in grammar.fromTerminal(c)
    push!(stack, (lhs, grammar.termination))
  end
end

function push_completions!(grammar::Grammar, stack, c::NonTerminal)
  if c in grammar.start_categories
    push!(stack, (initial_category(grammar), StartRule(c)))
  end
end

function push_completions!(
  grammar::Grammar, stack, c1::NonTerminal, c2::NonTerminal
)
  if c1 == c2
    if duplication(grammar) in grammar.nt_rules
      push!(stack, (c1, duplication(grammar)))
    end
  else
    if LeftHeaded(c2) in grammar.nt_rules
      push!(stack, (c1, LeftHeaded(c2)))
    end
    if RightHeaded(c1) in grammar.nt_rules
      push!(stack, (c2, RightHeaded(c1)))
    end
  end
end

function push_completions!(grammar::Grammar, stack, c1, c2)
  nothing
end

using Test
T, NT = String, Symbol
nt = NonTerminal{T,NT}(:bar)
LeftHeaded(nt)
lhr = LeftHeaded{T,NT}(NonTerminal{T,NT}(:foo))
@test lhr(nt) == map(NonTerminal{T,NT}, (:bar, :foo))

T, NT = String, Symbol
nonterminals = map(NonTerminal{T, NT}, [:a, :b, :c1, :c2])
terminals = map(Terminal{T, NT}, ["a", "b", "c"])
termination_dict = Dict(:a => "a", :b => "b", :c1 => "c", :c2 => "c")
toTerminal(c) = Terminal{T,NT}(termination_dict[c.label])
termination = Termination{T,NT,typeof(toTerminal)}(toTerminal)
inv_termination_dict = Dict("a" => [:a], "b" => [:b], "c" => [:c1, :c2])
fromTerminal(c) = map(NonTerminal{T, NT}, inv_termination_dict[c.label])
nt_rules = Set([ 
  Duplication{T,NT}(); 
  map(LeftHeaded, nonterminals); 
  map(RightHeaded, nonterminals)])
start_categories = Set([first(nonterminals)])

grammar = Grammar(nt_rules, start_categories, termination, fromTerminal)

terminalss = [[rand(terminals)] for _ in 1:100]
@time chart = chartparse(grammar, CountScoring(), terminalss)
chart[1,10][initial_category(grammar)]



T, NT = String, Symbol
nonterminals = map(NonTerminal{T, NT}, Symbol.(:a, 1:100))
terminals = map(Terminal{T, NT}, string.("a", 1:100))
toTerminal(c) = Terminal{T,NT}(string(c.label))
termination = Termination{T,NT,typeof(toTerminal)}(toTerminal)
fromTerminal(c) = [NonTerminal{T, NT}(Symbol(c.label))]
nt_rules = Set([ 
  Duplication{T,NT}(); 
  map(LeftHeaded, nonterminals); 
  map(RightHeaded, nonterminals)])
start_categories = Set(nonterminals)

grammar = Grammar(nt_rules, start_categories, termination, fromTerminal)

terminalss = [[rand(terminals)] for _ in 1:100]
@time chart = chartparse(grammar, CountScoring(), terminalss)
chart[1,100][initial_category(grammar)]





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
