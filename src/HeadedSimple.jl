module HeadedSimple

using ..AbstractGrammars
import ..AbstractGrammars: apply, push_completions!, default
import Distributions: logpdf

##################
### Categories ###
##################

@enum CategoryTag start terminal nonterminal
default(::Type{CategoryTag}) = start

struct Category{V}
  tag :: CategoryTag
  val :: V
end

default(::Type{Category{V}}) where V = Category(default(CategoryTag), default(V))

start_cat(V) = Category(start, default(V))
terminal_cat(val) = Category(terminal, val)
nonterminal_cat(val) = Category(nonterminal, val)

terminal_cat(c::Category) = Category(terminal, c.val)
nonterminal_cat(c::Category) = Category(nonterminal, c.val)

#############
### Rules ###
#############

@enum RuleTag startrule terminate duplicate leftheaded rightheaded
default(::Type{RuleTag}) = startrule

struct Rule{V} <: AbstractRule{Category{V}}
  tag :: RuleTag
  cat :: Category{V}
end

default(::Type{Rule{V}}) where V = Rule(default(RuleTag), default(Category{V}))

start_rule(c) = Rule(startrule, c)
termination_rule(V) = Rule(terminate, default(Category{V}))
duplication_rule(V) = Rule(duplicate, default(Category{V}))
leftheaded_rule(c) = Rule(leftheaded, c)
rightheaded_rule(c) = Rule(rightheaded, c)

function apply(r::Rule, c::Category)
  duplicate   ⊣ r               && return (c, c)
  leftheaded  ⊣ r && c != r.cat && return (c, r.cat)
  rightheaded ⊣ r && c != r.cat && return (r.cat, c)
  startrule  ⊣ r && start ⊣ c  && return (r.cat,)
  terminate   ⊣ r               && return (terminal_cat(c),)
  return nothing
end

################
### Grammars ###
################

struct Grammar{V,P} <: AbstractGrammar{Rule{V}}
  rules  :: Set{Rule{V}}
  params :: P
end

function push_completions!(
    grammar::Grammar{V,P}, stack::Vector, c::Category{V}
  ) where {V,P}

  if terminal ⊣ c
    push!(stack, App(nonterminal_cat(c), termination_rule(V)))
  elseif nonterminal ⊣ c && start_rule(c) in grammar.rules
    push!(stack, App(start_cat(V), start_rule(c)))
  end
end

function push_completions!(
    grammar::Grammar{V,P}, stack::Vector, c1::Category{V}, c2::Category{V}
  ) where {V,P}

  nonterminal ⊣ c1 && nonterminal ⊣ c2 || return nothing
  if c1 == c2
    r = duplication_rule(V)
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

# V = Int
# ts = terminal_cat.(1:4)
# nts = nonterminal_cat.(1:4)
# rules = Set([
#   start_rule(nts[1]);
#   termination_rule(V);
#   duplication_rule(V);
#   leftheaded_rule.(nts);
#   rightheaded_rule.(nts)])
# grammar = Grammar(rules, nothing)

# import Distributions: logpdf
# logpdf(::Grammar, lhs, rule) = log(0.5)

# seq = [[rand(ts)] for _ in 1:70]
# scoring = AbstractGrammars.WDS(grammar)
# @time chart = chartparse(grammar, scoring, seq)
# @time rule_samples = sample_derivations(scoring, chart[1,70][nts[1]], 100)

end # module