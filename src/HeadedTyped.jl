module HeadedTyped

using Test # for development

using ..AbstractGrammars
import ..AbstractGrammars: apply, push_completions!, default
import Distributions: logpdf

##################
### Categories ###
##################

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

#############
### Rules ###
#############

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

struct Grammar{T,NT,TT,FT,P} <: AbstractGrammar{Rule{T,NT}}
  rules        :: Set{Rule{T,NT}}
  toTerminal   :: TT # function NonTerminal{T,NT} -> Terminal{T,NT}
  fromTerminal :: FT # function Terminal{T,NT} -> Vector{NonTerminal{T,NT}}
  params       :: P
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
      push!(stack, App(lhs, termination_rule(grammar)))
    end
  elseif nonterminal ⊣ c && start_rule(c) in grammar.rules
    push!(stack, App(initial_category(grammar), start_rule(c)))
  end
end

function push_completions!(grammar::Grammar, stack, c1, c2)
  nonterminal ⊣ c1 && nonterminal ⊣ c2 || return nothing
  if c1 == c2
    r = duplication_rule(grammar)
    if r in grammar.rules
      push!(stack, App(c1, r))
    end
  else
    r = leftheaded_rule(c2)
    if r in grammar.rules
      push!(stack, App(c1, r))
    end
    r = rightheaded_rule(c1)
    if r in grammar.rules
      push!(stack, App(c2, r))
    end
  end
end

#############
### Tests ###
#############

# T, NT = Float64, Int
# nt = nonterminal_category(T, 1)
# lhr = leftheaded_rule(nonterminal_category(T, 2))
# grammar = Grammar(Set([lhr]), nothing, nothing)
# @test apply(grammar, lhr, nt) == nonterminal_category.(T, (1, 2))



# T, NT = Int, Int
# nonterminals = nonterminal_category.(T, 1:4)
# terminals = terminal_category.(NT, 1:3)
# termination_dict = Dict(1 => 1, 2 => 2, 3 => 3, 4 => 3)
# toTerminal(c) = terminal_category(NT, termination_dict[c.ntlabel])
# inv_termination_dict = Dict(1 => [1], 2 => [2], 3 => [3, 4])
# fromTerminal(c) = nonterminal_category.(T, inv_termination_dict[c.tlabel])
# rules = Set{Rule{T,NT}}([
#   start_rule(first(nonterminals));
#   termination_rule(T, NT);
#   duplication_rule(T, NT); 
#   # leftheaded_rule.(nonterminals); 
#   rightheaded_rule.(nonterminals)])
# grammar = Grammar(rules, toTerminal, fromTerminal)

# initial_category(grammar)

# nonterminals .|> println
# rules .|> println

# struct Params{F <: AbstractFloat, N}
#   start :: Vector{F}
#   nt    :: NTuple{N, Vector{F}}
# end



# function logpdf(::Grammar)

# end


# logpdf(::Grammar, lhs, rule) = log(0.5)
# logpdf(grammar::Grammar, app::App) = logpdf(grammar, app.lhs, app.rule)

# terminalss = [[rand(terminals)] for _ in 1:70]
# scoring = AbstractGrammars.WDS(grammar)
# @time chart = chartparse(grammar, scoring, terminalss)
# @time rule_samples = sample_derivations(scoring, chart[1,70][initial_category(grammar)], 100)
# @time sum(logpdf(grammar, r) for r in rule_samples)


# T, NT = Float64, Int
# nonterminals = nonterminal_category.(T, 1:100)
# terminals = terminal_category.(NT, 1:100)
# toTerminal(c) = terminal_category(NT, float(c.ntlabel))
# fromTerminal(c) = [nonterminal_category(T, Int(c.tlabel))]
# rules = Set{Rule{T,NT}}([
#   start_rule.(nonterminals);
#   termination_rule(T, NT);
#   duplication_rule(T, NT); 
#   # leftheaded_rule.(nonterminals); 
#   rightheaded_rule.(nonterminals)])
# grammar = Grammar(rules, toTerminal, fromTerminal)


# terminalss = [[rand(terminals)] for _ in 1:100]
# @time chart = chartparse(grammar, CountScoring(), terminalss)
# chart[1,100][initial_category(grammar)]
# scoring = AbstractGrammars.WDS(grammar)
# @time chart = chartparse(grammar, scoring, terminalss)
# sample_derivations(scoring, chart[1,100][initial_category(grammar)], 1)
# s
# scoring.store
# eltype(scoring.store) |> isbitstype

# @time chart = chartparse(grammar, AbstractGrammars.TreeDistScoring(), terminalss)



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
