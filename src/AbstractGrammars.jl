module AbstractGrammars

# Imports
# using Distributions
# using LogProbs

# using SpecialFunctions: logbeta
# using MacroTools: @capture

# import Base: rand, minimum, maximum
# import Distributions: sampler, logpdf, cdf, quantile
# import StatsBase: params

export AbstractRule, 
       AbstractGrammar, 
       initial_category,
       push_completions!
export Scoring,
       score_type,
       calc_score,
       InsideScoring,
       CountScoring
export chartparse

#############
### Utils ###
#############

normalize(xs) = xs ./ sum(xs)

########################
### Conjugate Models ###
########################

include("ConjugateModels.jl")

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

################
### Scorings ###
################

struct InsideScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{InsideScoring}) = LogProb

function calc_score(grammar::AbstractGrammar, ::InsideScoring, lhs, rule)
  LogProb(logpdf(grammar.ruledist, lhs, rule))
end

struct CountScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{CountScoring}) = Int

function calc_score(grammar::AbstractGrammar, ::CountScoring, lhs, rule)
  1
end

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
function insert!(category::C, score::S; into::ChartCell{C, S}) where {C, S}
  chart_cell = into
  if haskey(chart_cell, category)
    chart_cell[category] += score
  else
    chart_cell[category]  = score
  end
end

function chartparse(grammar::G, scoring, terminalss::Vector{Vector{C}}) where
  {C, R <: AbstractRule{C}, G <: AbstractGrammar{R}}
  n = length(terminalss) # sequence length
  S = score_type(grammar, scoring)
  chart = empty_chart(C, S, n)
  stack = Vector{Tuple{C, R}}() # channel for communicating completions
  # using a single stack is much more efficient than constructing multiple arrays

  score(lhs, rule) = calc_score(grammar, scoring, lhs, rule)

  for (i, terminals) in enumerate(terminalss)
    for terminal in terminals
      push_completions!(grammar, stack, terminal)
      while !isempty(stack)
        (lhs, rule) = pop!(stack)
        insert!(lhs, score(lhs, rule), into=chart[i, i])
      end
    end
  end

  for l in 1:n-1 # length
    for i in 1:n-l # start index
      j = i + l # end index
      for k in i:j-1 # split index
        for (rhs1, s1) in chart[i, k]
          for (rhs2, s2) in chart[k+1, j]
            push_completions!(grammar, stack, rhs1, rhs2)
            while !isempty(stack)
              (lhs, rule) = pop!(stack)
              insert!(lhs, score(lhs, rule) * s1 * s2, into=chart[i, j])
            end
          end
        end
      end
    end
  end

  chart
end

###############################
### Generic headed grammars ###
###############################

# @enum HeadedCategoryTag start_category terminal nonterminal

# struct GenericCategory{T}
#   isterminal :: Bool
#   label :: T
# end

# @enum HeadedRuleKind leftheaded rightheaded duplication start_rule termination

# struct HeadedRule{T} <: AbstractRule{GenericCategory{T}}
#   kind :: HeadedRuleKind
#   dependent :: T
# end

# function (r::HeadedRule{T})(c::GenericCategory{})
  
# end

# Union{Nothing, Int} |> isbitstype




end # module
