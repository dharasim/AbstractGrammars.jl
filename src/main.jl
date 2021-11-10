##########################
### Category interface ###
##########################

# abstract type AbstractCategory end

# """
#     isstart(category::AbstractCategory) ::Bool
# """
# function isstart end

# """
#     isterminal(category::AbstractCategory) ::Bool
# """
# function isterminal end

# """
#     isnonterminal(category::AbstractCategory) ::Bool

# The default implementation is: `isnonterminal(c) = !isterminal(c)`
# """
# function isnonterminal(category::AbstractCategory)
#   !isterminal(category)
# end

# possible tags: start, terminal, nonterminal, default
struct StdCategory{T}
  tag :: Tag
  val :: T

  function StdCategory(tag, val::T) where T 
    @assert tag in ("terminal", "nonterminal")
    new{T}(tag, val)
  end
  function StdCategory(tag, T::Type) 
    @assert tag in ("start", "default")
    new{T}(tag)
  end
end

isstart(c::StdCategory) = "start" ⊣ c
isterminal(c::StdCategory) = "terminal" ⊣ c

default(::Type{StdCategory{T}}) where T = StdCategory("default", T)

start_category(T::Type) = StdCategory("start", T)
terminal_category(val) = StdCategory("terminal", val)
nonterminal_category(val) = StdCategory("nonterminal", val)

terminal_category(c::StdCategory) = StdCategory("terminal", c.val)
nonterminal_category(c::StdCategory) = StdCategory("nonterminal", c.val)

######################
### Rule interface ###
######################

abstract type AbstractRule{Category} end
abstract type AbstractGrammar{Rule<:AbstractRule} end

"""
    arity(rule::AbstractRule) ::Int

Rules have constant length of their right-hand sides and 
`arity(rule)` returns this length.
"""
function arity end

"""
    apply(grammar, rule, category)

Apply `rule` to `category`, potentially using information from `grammar`.
Returns `nothing` if `rule` is not applicable to `category`.
Default implementation doesn't use `grammar`'s information and calls
`apply(rule, category)`.
"""
function apply(grammar, rule, category) 
  apply(rule, category)
end

# Rule applications
struct App{C, R <: AbstractRule{C}}
  lhs  :: C
  rule :: R
end

apply(grammar, app::App) = apply(grammar, app.rule, app.lhs)

using AbstractGrammars.AtMosts: AtMost, atmost2

struct StdRule{C} <: AbstractRule{C}
  lhs :: C
  rhs :: AtMost{C, 2}
end

StdRule(lhs, rhs...) = StdRule(lhs, atmost2(rhs...))
-->(lhs::C, rhs::C) where C = StdRule(lhs, rhs)
-->(lhs::C, rhs) where C = StdRule(lhs, rhs...)

apply(r::StdRule{C}, c::C) where C = r.lhs == c ? tuple(r.rhs...) : nothing
arity(r::StdRule) = length(r.rhs)

function show(io::IO, r::StdRule{C}) where C
  print(io, "StdRule{$C}($(r.lhs) -->")
  foreach(c -> print(io, " $c"), r.rhs)
  print(io, ")")
end

#########################
### Grammar interface ###
#########################

"""
    push_completions!(grammar, stack, c1[, c2])

Push all unary completions of `c1` or all binary completions of `(c1, c2)` 
on `stack`. Completions are typed as rule applications.
"""
function push_completions! end

"""
    logpdf(grammar, lhs, rule)

Logarithm of the probability of applying `rule` to `lhs`.
Parameters are typically contained in `grammar`.
"""
function logpdf end

mutable struct StdGrammar{C, P} <: AbstractGrammar{StdRule{C}}
  start       :: Set{C}
  rules       :: Set{StdRule{C}}
  completions :: Dict{AtMost{C, 2}, Vector{C}}
  params      :: P

  function StdGrammar(
      start, rules::Set{StdRule{C}}, params::P
    ) where {C, P}

    completions = Dict{AtMost{C, 2}, Vector{C}}()
    for r in rules
      comps = get!(() -> C[], completions, r.rhs)
      push!(comps, r.lhs)
    end
    return new{C, P}(Set(collect(start)), rules, completions, params)
  end
end

function push_completions!(grammar::StdGrammar, stack, categories...)
  rhs = atmost2(categories...)
  if haskey(grammar.completions, rhs)
    for lhs in grammar.completions[rhs]
      push!(stack, App(lhs, StdRule(lhs, rhs...)))
    end
  end
end

#########################
### Scoring interface ###
#########################

abstract type Scoring end

function score_type(grammar::AbstractGrammar, scoring::Scoring) 
  score_type(typeof(grammar), typeof(scoring))
end

function score_type(::Type{G}, ::Type{S}) where 
  {G <: AbstractGrammar, S <: Scoring}
  error("Function score_type not implemented for $G with $S.")
end

function ruleapp_score(scoring, grammar, lhs, rule)
  error("Function ruleapp_score not implemented")
end

function add_scores(scoring, left, right) 
  error("Function add_scores not implemented") 
end

function mul_scores(scoring, left, right)
  error("Function mul_scores not implemented") 
end

mul_scores(scoring::Scoring, s1, s2, s3) = 
  mul_scores(scoring, s1, mul_scores(scoring, s2, s3))