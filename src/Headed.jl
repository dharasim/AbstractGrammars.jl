module Headed

export Grammar, Rule, start_rule, termination_rule, duplication_rule, leftheaded_rule, 
  rightheaded_rule

using ..AbstractGrammars: AbstractRule, AbstractGrammar, Tag, App
using ..AbstractGrammars.GeneralCategories

import ..AbstractGrammars: apply, push_completions!, default
import Distributions:logpdf

#############
### Rules ###
#############

# possible tags: start, terminate, duplicate, leftheaded, rightheaded, default
struct Rule{T} <: AbstractRule{Category{T}}
    tag::Tag
    cat::Category{T}

    function Rule(tag, c::Category{T}) where T 
        @assert tag in ("start", "leftheaded", "rightheaded")
        new{T}(tag, c)
    end
    function Rule(tag, T::Type) 
        @assert tag in ("terminate", "duplicate", "default")
        new{T}(tag)
    end
end

default(::Type{Rule{T}}) where T = Rule("default", T)

start_rule(c::Category) = Rule("start", c)
termination_rule(T::Type) = Rule("terminate", T)
duplication_rule(T::Type) = Rule("duplicate", T)
leftheaded_rule(c::Category) = Rule("leftheaded", c)
rightheaded_rule(c::Category) = Rule("rightheaded", c)

function apply(r::Rule, c::Category)
    "duplicate"   ⊣ r                && return (c, c)
    "leftheaded"  ⊣ r && c != r.cat  && return (c, r.cat)
    "rightheaded" ⊣ r && c != r.cat  && return (r.cat, c)
    "start"       ⊣ r && "start" ⊣ c && return (r.cat,)
    "terminate"   ⊣ r                && return (terminal_category(c),)
    "default"     ⊣ r                && error("default rule cannot be applied")
    return nothing
end

################
### Grammars ###
################

struct Grammar{T,P} <: AbstractGrammar{Rule{T}}
    rules::Set{Rule{T}}
    params::P
end

function push_completions!(
    grammar::Grammar{T,P}, stack::Vector, c::Category{T}
  ) where {T,P}

    if "terminal" ⊣ c
        push!(stack, App(nonterminal_category(c), termination_rule(T)))
    elseif "nonterminal" ⊣ c && start_rule(c) in grammar.rules
    push!(stack, App(start_category(T), start_rule(c)))
    end
end

function push_completions!(
    grammar::Grammar{T,P}, stack::Vector, c1::Category{T}, c2::Category{T}
  ) where {T,P}

    "nonterminal" ⊣ c1 && "nonterminal" ⊣ c2 || return nothing
    if c1 == c2
        r = duplication_rule(T)
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
