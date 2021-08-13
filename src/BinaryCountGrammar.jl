module BinaryCountGrammar

using  ..AbstractGrammars
import ..AbstractGrammars: initial_category, push_completions!

export test_binary_count_grammar

@enum Category nonterminal terminal

# rules can either branch or terminate
@enum RuleKind branch terminate
struct Rule <: AbstractRule{Category}
  kind :: RuleKind
end

function (r::Rule)(lhs::Category)
  if lhs == nonterminal
    if r.kind == branch
      (nonterminal, nonterminal)
    else
      (terminal,)
    end
  else
    nothing
  end
end

struct Grammar <: AbstractGrammar{Rule} end
initial_category(::Grammar) = nonterminal

function push_completions!(::Grammar, stack, c)
  if c == terminal
    push!(stack, (nonterminal, Rule(terminate)))
  end
end

function push_completions!(::Grammar, stack, c1, c2)
  if c1 == nonterminal && c2 == nonterminal
    push!(stack, (nonterminal, Rule(branch)))
  end
end

using Test
function test_binary_count_grammar()
  terminalss = fill([terminal], 10)
  grammar = Grammar()
  @time chart = chartparse(grammar, CountScoring(), terminalss)
  @test [chart[1,k][initial_category(grammar)] for k in 1:10] ==
        [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]
end

end # module