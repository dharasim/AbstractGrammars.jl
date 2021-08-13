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