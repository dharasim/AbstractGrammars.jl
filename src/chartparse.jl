const ChartCell{Category,Score} = Dict{Category,Score}
const Chart{Category,Score} = Matrix{ChartCell{Category,Score}}

function empty_chart(::Type{Category}, ::Type{Score}, n) where {Category,Score}
    [ Dict{Category,Score}() for _ in 1:n, _ in 1:n ]
end

function insert!(chart_cell::ChartCell, scoring, category, score::S) where S
    s = get(chart_cell, category, zero(S))
    chart_cell[category] = add_scores(scoring, s, score)
end

struct ScoredCategory{C,S}
    category::C
    score::S
end

function chartparse(grammar, scoring, terminals)
    terminalss = [[t] for t in terminals]
    chartparse(grammar, scoring, terminalss)
end

function chartparse(
        grammar::G, scoring, terminalss::Vector{Vector{C}}
    ) where {C, R <: Rule{C}, G <: Grammar{R}}

    n = length(terminalss) # sequence length
    S = scoretype(scoring, grammar)
    chart = empty_chart(C, S, n)
    stack = Vector{App{C,R}}() # channel for communicating completions
    # using a single stack is much more efficient than constructing multiple arrays
    stack_unary = Vector{ScoredCategory{C,S}}()

    score(app) = ruleapp_score(scoring, app.lhs, app.rule)

    for (i, terminals) in enumerate(terminalss)
        for terminal in terminals
            push_completions!(grammar, stack, terminal)
            while !isempty(stack)
                app = pop!(stack)
                insert!(chart[i, i], scoring, app.lhs, score(app))
            end
        end
    end

    for l in 1:n - 1 # length
        for i in 1:n - l # start index
            j = i + l # end index

      # binary completions
            for k in i:j - 1 # split index
                for (rhs1, s1) in chart[i, k]
                    for (rhs2, s2) in chart[k + 1, j]
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
