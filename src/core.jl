using LogProbs
using Distributions: logpdf

#############
### Utils ###
#############

normalize(xs) = xs ./ sum(xs)

function isplain(T::Type) 
  v = Vector{T}(undef, 1)
  try
    first(v)
  catch e
    return false
  end
  return true
end

# check for the tag of an object
⊣(tag, x) = x.tag == tag

default(::Type{T}) where T <: Number = zero(T)
default(::Type{Symbol}) = Symbol()

#########################
### Grammar Interface ###
#########################

abstract type AbstractRule{Category} end

abstract type AbstractGrammar{Rule<:AbstractRule} end

function initial_category(grammar::G) where G<:AbstractGrammar
  error("Function initial_category not implemented for $G.")
end

function push_completions!(grammar::G, stack, c) where G<:AbstractGrammar
  error("Function push_completions! (unary) not implemented for $G.")
end

function push_completions!(grammar::G, stack, c1, c2) where G<:AbstractGrammar
  error("Function push_completions! (binary) not implemented for $G.")
end

# rule application
struct App{C, R <: AbstractRule{C}}
  lhs  :: C
  rule :: R
end

function App(::AbstractGrammar{R}, lhs, rule) where 
  {C, R <: AbstractRule{C}}
  App{C, R}(lhs, rule)
end

apply(::AbstractGrammar, rule, category) = apply(rule, category)
apply(grammar::AbstractGrammar, app::App) = apply(grammar, app.rule, app.lhs)


# function logpdf(grammar, lhs, rule) end
# function rand_rule(grammar, lhs) end
# function observe_rule!(grammar, lhs, rule, pseudocount) end

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

######################
### Inside scoring ###
######################

struct InsideScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{InsideScoring}) = LogProb
ruleapp_score(::InsideScoring, grammar::AbstractGrammar, lhs, rule) =
  LogProb(logpdf(grammar, lhs, rule), islog=true)
add_scores(::InsideScoring, left, right) = left + right
mul_scores(::InsideScoring, left, right) = left * right

#####################
### Count scoring ###
#####################

struct CountScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{CountScoring}) = Int
ruleapp_score(::CountScoring, ::AbstractGrammar, ::Any, ::Any) = 1
add_scores(::CountScoring, left, right) = left + right
mul_scores(::CountScoring, left, right) = left * right

#######################
### Boolean scoring ###
#######################

struct BooleanScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{BooleanScoring}) = Bool
ruleapp_score(::AbstractGrammar, ::BooleanScoring, ::Any, ::Any) = true
add_scores(::BooleanScoring, left, right) = left || right
mul_scores(::BooleanScoring, left, right) = left && right

######################################################################
### Free-semiring scorings with manually managed pointer structure ###
######################################################################

# Implementation idea: break rec. structure with indices into a vector (store).
# Ihe store contains unboxed values, which reduces GC times.
# Additionally, it allows to update probabilities without parsing again (not yet implemented).

@enum ScoreTag ADD MUL VAL ZERO

struct ScoredFreeEntry{S,T}
  tag        :: ScoreTag
  score      :: S
  value      :: T
  index      :: Int
  leftIndex  :: Int
  rightIndex :: Int

  # addition and multiplication
  function ScoredFreeEntry(
    store :: Vector{ScoredFreeEntry{S,T}},
    op    :: Union{typeof(+), typeof(*)},
    left  :: ScoredFreeEntry{S,T}, 
    right :: ScoredFreeEntry{S,T}
  ) where {S,T}
    tag(::typeof(+)) = ADD
    tag(::typeof(*)) = MUL
    score = op(left.score, right.score)
    value = left.value # dummy value
    index = length(store) + 1
    x = new{S,T}(tag(op), score, value, index, left.index, right.index)
    push!(store, x)
    return x
  end

  # scored values
  function ScoredFreeEntry(
    store :: Vector{ScoredFreeEntry{S,T}},
    score :: S,
    value :: T
  ) where {S,T}
    index = length(store) + 1
    x = new{S,T}(VAL, score, value, index)
    push!(store, x)
    return x
  end

  # constant zero
  function ScoredFreeEntry(::Type{S}, ::Type{T}) where {S,T}
    new{S,T}(ZERO, zero(S))
  end
end

import Base: zero, iszero
zero(::Type{ScoredFreeEntry{S,T}}) where {S,T} = ScoredFreeEntry(S, T)
iszero(x::ScoredFreeEntry) = x.tag == ZERO

# Weighted Derivation Scoring (WDS)
struct WDS{S,T} <: Scoring
  store :: Vector{ScoredFreeEntry{S,T}}
end

function WDS(::G) where
  {C, R <: AbstractRule{C}, G <: AbstractGrammar{R}}
  WDS(ScoredFreeEntry{LogProb, App{C, R}}[])
end

score_type(::Type{<:AbstractGrammar}, ::Type{<:WDS{S,T}}) where {S, T} =
  ScoredFreeEntry{S, T}
ruleapp_score(s::WDS, grammar, lhs, rule) = 
  ScoredFreeEntry(
    s.store, 
    LogProb(logpdf(grammar, lhs, rule), islog=true), 
    App(lhs, rule)
  )

function add_scores(s::WDS, x, y)
  ZERO == x.tag && return y
  ZERO == y.tag && return x
  return ScoredFreeEntry(s.store, +, x, y)
end

function mul_scores(s::WDS, x, y)
  ZERO == x.tag && return x
  ZERO == y.tag && return y
  return ScoredFreeEntry(s.store, *, x, y)
end

function sample_derivations(s::WDS{S,T}, x::ScoredFreeEntry{S,T}, n::Int) where {S,T}
  vals = Vector{T}()
  for _ in 1:n
    sample_derivation!(vals, s, x)
  end
  vals
end

function sample_derivation!(vals, s::WDS, x::ScoredFreeEntry{S,T}) where {S,T}
  if VAL  ⊣ x 
    push!(vals, x.value)
  elseif ADD ⊣ x
    index = rand(S) < s.store[x.leftIndex].score / x.score ? x.leftIndex : x.rightIndex
    sample_derivation!(vals, s, s.store[index])
  elseif MUL ⊣ x
    sample_derivation!(vals, s, s.store[x.leftIndex])
    sample_derivation!(vals, s, s.store[x.rightIndex])
  else # ZERO ⊣ x
    error("cannot sample from zero")
  end
end

########################################################
### Free-semiring scorings with re-computable scores ###
########################################################

# To be done

#####################
### Chart parsing ###
#####################

import Base: insert!

const ChartCell{Category, Score} = Dict{Category, Score}
const Chart{Category, Score} = Matrix{ChartCell{Category, Score}}

function empty_chart(::Type{Category}, ::Type{Score}, n) where {Category, Score}
  [ Dict{Category, Score}() for i in 1:n, j in 1:n ]
end

"""
    insert!(category, score, into=chart_cell)
"""
function insert!(chart_cell::ChartCell, scoring, category, score::S) where S
  s = get(chart_cell, category, zero(S))
  chart_cell[category] = add_scores(scoring, s, score)
end

struct ScoredCategory{C, S}
  category :: C
  score    :: S
end

function chartparse(grammar::G, scoring, terminalss) where {
  C, R <: AbstractRule{C}, G <: AbstractGrammar{R}
}
  n = length(terminalss) # sequence length
  S = score_type(grammar, scoring)
  chart = empty_chart(C, S, n)
  stack = Vector{App{C, R}}() # channel for communicating completions
  # using a single stack is much more efficient than constructing multiple arrays
  stack_unary = Vector{ScoredCategory{C, S}}()

  score(app) = ruleapp_score(scoring, grammar, app.lhs, app.rule)

  for (i, terminals) in enumerate(terminalss)
    for terminal in terminals
      push_completions!(grammar, stack, terminal)
      while !isempty(stack)
        app = pop!(stack)
        insert!(chart[i, i], scoring, app.lhs, score(app))
      end
    end
  end

  for l in 1:n-1 # length
    for i in 1:n-l # start index
      j = i + l # end index
      # binary completions
      for k in i:j-1 # split index
        for (rhs1, s1) in chart[i, k]
          for (rhs2, s2) in chart[k+1, j]
            push_completions!(grammar, stack, rhs1, rhs2)
            while !isempty(stack)
              app = pop!(stack)
              s = mul_scores(scoring, score(app), s1, s2)
              insert!(chart[i, j], scoring, app.lhs, s)
            end
          end
        end
      end

      # unary completions
      for (rhs, s) in chart[i, j]
        push_completions!(grammar, stack, rhs)
        while !isempty(stack)
          app = pop!(stack)
          push!(stack_unary, 
            ScoredCategory{C,S}(app.lhs, mul_scores(scoring, score(app), s)))
        end
      end
      while !isempty(stack_unary)
        sc = pop!(stack_unary) # pop a scored category
        insert!(chart[i, j], scoring, sc.category, sc.score)
      end
    end
  end

  return chart
end