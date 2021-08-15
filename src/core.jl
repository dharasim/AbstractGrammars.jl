#############
### Utils ###
#############

normalize(xs) = xs ./ sum(xs)

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

function calc_score(grammar::G, scoring::S, lhs, rule) where 
  {G <: AbstractGrammar, S <: Scoring}
  error("Function calc_score not implemented for $G with $S.")
end

# imports for scoring implementations
# each score type must be a semiring
import Base: zero, iszero, one, isone, +, * 

######################
### Inside scoring ###
######################

struct InsideScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{InsideScoring}) = LogProb
function calc_score(grammar::AbstractGrammar, ::InsideScoring, lhs, rule)
  LogProb(logpdf(grammar, lhs, rule))
end

#####################
### Count scoring ###
#####################

struct CountScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{CountScoring}) = Int
calc_score(::AbstractGrammar, ::CountScoring, ::Any, ::Any) = 1

#######################
### Boolean scoring ###
#######################

struct BoolScore value::Bool end
zero(::BoolScore) = BoolScore(false)
zero(::Type{BoolScore}) = BoolScore(false)
iszero(x::BoolScore) = x.value === false
one(::BoolScore) = BoolScore(true)
one(::Type{BoolScore}) = BoolScore(true)
isone(x::BoolScore) = x.value === true
+(x::BoolScore, y::BoolScore) = BoolScore(x.value || y.value)
*(x::BoolScore, y::BoolScore) = BoolScore(x.value && y.value)

struct BooleanScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{BooleanScoring}) = BoolScore
calc_score(::AbstractGrammar, ::BooleanScoring, ::Any, ::Any) = one(BoolScore)

##############################
### Free-semiring scorings ###
##############################

abstract type ScoredFree{S,T} end
struct Zero{S,T} <: ScoredFree{S,T} end
struct One{S,T} <: ScoredFree{S,T} end
struct Value{S,T} <: ScoredFree{S,T} score::S; value::T end
struct Add{S,T} <: ScoredFree{S,T}
  score::S
  left::ScoredFree{S,T}
  right::ScoredFree{S,T}
end
struct Mul{S,T} <: ScoredFree{S,T}
  score::S
  left::ScoredFree{S,T}
  right::ScoredFree{S,T}
end

import Base: show
show(io::IO, x::Add{S,T}) where {S,T} =
  print(io, "Add{$S, $T}($(x.score), ...left, ...right)")
show(io::IO, x::Mul{S,T}) where {S,T} =
  print(io, "Mul{$S, $T}($(x.score), ...left, ...right)")

zero(::ScoredFree{S,T}) where {S,T} = Zero{S,T}()
zero(::Type{<:ScoredFree{S,T}}) where {S,T} = Zero{S,T}()
iszero(::Zero) = true
iszero(::ScoredFree) = false
one(::ScoredFree{S,T}) where {S,T} = One{S,T}()
one(::Type{<:ScoredFree{S,T}}) where {S,T} = One{S,T}()
isone(::One) = true
isone(::ScoredFree) = false

# normal addition and multiplication
+(x::ScoredFree, y::ScoredFree) = Add(x.score+y.score, x, y)
*(x::ScoredFree, y::ScoredFree) = Mul(x.score*y.score, x, y)

# Zero is neutral element for addition
+(::Zero, x::ScoredFree) = x
+(x::ScoredFree, ::Zero) = x
+(x::Zero, ::Zero) = x

# One is neutral element for multiplication
*(::One, x::ScoredFree) = x
*(x::ScoredFree, ::One) = x
*(x::One, ::One) = x

# Zero absorbs anything in multiplication
*(x::Zero, ::ScoredFree) = x
*(x::Zero, ::One) = x
*(::ScoredFree, x::Zero) = x
*(::One, x::Zero) = x
*(x::Zero, ::Zero) = x

function sample_monom(sf::ScoredFree{S,T}) where {S,T}
  vals = Vector{T}()
  sample_monom!(vals, sf)
end

sample_monom!(::Any, ::Zero) = error("cannot sample from zero")
sample_monom!(::Any, ::One) = nothing
sample_monom!(vals, v::Value) = push!(vals, v.value)
sample_monom!(vals, a::Add{S,T}) where {S,T} = 
  sample_monom!(vals, rand(S) < a.left.score / a.score ? a.left : a.right)
sample_monom!(vals, m::Mul) =
  (sample_monom!(vals, m.left); sample_monom!(vals, m.right))

struct CompactForrestScoring <: Scoring end
score_type(::Type{<:AbstractGrammar{R}}, ::Type{CompactForrestScoring}) where 
  {C, R <: AbstractRule{C}} = ScoredFree{BoolScore, Tuple{C, R}}
calc_score(::AbstractGrammar{R}, ::CompactForrestScoring, lhs, rule) where
  {C, R <: AbstractRule{C}} = 
    Value{BoolScore, Tuple{C, R}}(one(BoolScore), (lhs, rule))

struct TreeDistScoring <: Scoring end
score_type(::Type{<:AbstractGrammar{R}}, ::Type{TreeDistScoring}) where 
  {C, R <: AbstractRule{C}} = ScoredFree{LogProb, Tuple{C, R}}
calc_score(grammar::AbstractGrammar{R}, ::TreeDistScoring, lhs, rule) where
  {C, R <: AbstractRule{C}} =
    Value{LogProb, Tuple{C,R}}(LogProb(logpdf(grammar, lhs, rule)), (lhs, rule))

using Test
S, T = Int, Symbol
@test zero(One{S,T}) === zero(One{S,T}()) === Zero{S,T}()
@test one(Zero{S,T}) === one(Zero{S,T}()) === One{S,T}()

######################################################################
### Free-semiring scorings with manually managed pointer structure ###
######################################################################

@enum ScoreTag ADD MUL VAL

struct ScoredFreeStored{S,T}
  store      :: Vector{ScoredFreeStored{S,T}}
  tag        :: ScoreTag
  score      :: S
  value      :: T
  index      :: Int
  leftIndex  :: Int
  rightIndex :: Int

  function ScoredFreeStored(
    store :: Vector{ScoredFreeStored{S,T}}, 
    op    :: Union{typeof(+), typeof(*)}, 
    left  :: ScoredFreeStored{S,T}, 
    right :: ScoredFreeStored{S,T}
  ) where {S,T}
    tag(::typeof(+)) = ADD
    tag(::typeof(*)) = MUL
    score = op(left.score, right.score)
    value = left.value # dummy value
    index = length(store) + 1
    x = new{S,T}(store, tag(op), score, value, index, left.index, right.index)
    push!(store, x)
    x
  end

  function ScoredFreeStored(
    store :: Vector{ScoredFreeStored{S,T}},
    score :: S,
    value :: T
  ) where {S,T}
    index = length(store) + 1
    x = new{S,T}(store, VAL, score, value, index, -1, -1)
    push!(store, x)
    x
  end
end

struct Foo{T}
  ts :: Vector{T}
  t  :: T
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
function insert!(chart_cell::ChartCell, category, score)
  if haskey(chart_cell, category)
    chart_cell[category] += score
  else
    chart_cell[category]  = score
  end
end

function chartparse(grammar::G, scoring, terminalss) where {
  C, R <: AbstractRule{C}, G <: AbstractGrammar{R}
}
  n = length(terminalss) # sequence length
  S = score_type(grammar, scoring)
  chart = empty_chart(C, S, n)
  stack = Vector{Tuple{C, R}}() # channel for communicating completions
  # using a single stack is much more efficient than constructing multiple arrays
  stack_unary = Vector{Tuple{C, S}}()

  score(lhs, rule) = calc_score(grammar, scoring, lhs, rule)

  for (i, terminals) in enumerate(terminalss)
    for terminal in terminals
      push_completions!(grammar, stack, terminal)
      while !isempty(stack)
        (lhs, rule) = pop!(stack)
        insert!(chart[i, i], lhs, score(lhs, rule))
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
              (lhs, rule) = pop!(stack)
              insert!(chart[i, j], lhs, score(lhs, rule) * s1 * s2)
            end
          end
        end
      end

      # unary completions
      for (rhs, s) in chart[i, j]
        push_completions!(grammar, stack, rhs)
        while !isempty(stack)
          (lhs, rule) = pop!(stack)
          push!(stack_unary, (lhs, score(lhs, rule) * s))
        end
      end
      while !isempty(stack_unary)
        (lhs, s) = pop!(stack_unary)
        insert!(chart[i, j], lhs, s)
      end
    end
  end

  chart
end

function chartparse_optimized(grammar::G, scoring, terminalss) where {
  C, R <: AbstractRule{C}, G <: AbstractGrammar{R}
}
  n = length(terminalss) # sequence length
  S = score_type(grammar, scoring)
  chart = empty_chart(C, S, n)
  stack = Vector{Tuple{C, R}}() # channel for communicating completions
  # using a single stack is much more efficient than constructing multiple arrays
  stack_unary = Vector{Tuple{C, S}}()

  score(lhs, rule) = calc_score(grammar, scoring, lhs, rule)

  for (i, terminals) in enumerate(terminalss)
    for terminal in terminals
      push_completions!(grammar, stack, terminal)
      while !isempty(stack)
        (lhs, rule) = pop!(stack)
        insert!(chart[i, i], lhs, score(lhs, rule))
      end
    end
  end

  for l in 1:n-1 # length
    for i in 1:n-l # start index
      j = i + l # end index

      # binary completions
      for k in i:j-1 # split index
        for (rhs1, s1) in chart[i, k]
          # rhs1 = rhs1_s1[1]
          # s1   = rhs1_s1[2]
          for (rhs2, s2) in chart[k+1, j]
            # rhs2 = rhs2_s2[1]
            # s2   = rhs2_s2[2]
            push_completions!(grammar, stack, rhs1, rhs2)
            while !isempty(stack)
              lhs_rule = pop!(stack)
              lhs  = lhs_rule[1]
              rule = lhs_rule[2]
              insert!(chart[i, j], lhs, score(lhs, rule) * s1 * s2)
            end
          end
        end
      end

      # unary completions
      for (rhs, s) in chart[i, j]
        push_completions!(grammar, stack, rhs)
        while !isempty(stack)
          lhs_rule = pop!(stack)
            lhs  = lhs_rule[1]
            rule = lhs_rule[2]
          push!(stack_unary, (lhs, score(lhs, rule) * s))
        end
      end
      while !isempty(stack_unary)
        lhs_s = pop!(stack_unary)
        lhs = lhs_s[1]
        s   = lhs_s[2]
        insert!(chart[i, j], lhs, s)
      end
    end
  end

  chart
end