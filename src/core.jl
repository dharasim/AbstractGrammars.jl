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

######################
### Inside scoring ###
######################

struct InsideScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{InsideScoring}) = LogProb

function calc_score(grammar::AbstractGrammar, ::InsideScoring, lhs, rule)
  LogProb(logpdf(grammar.ruledist, lhs, rule))
end

#####################
### Count scoring ###
#####################

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