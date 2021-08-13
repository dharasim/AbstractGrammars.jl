module AbstractGrammars

export 
# utils
normalize,

# Rule and grammar interface
AbstractRule, AbstractGrammar, initial_category, push_completions!,

# Scorings
Scoring, score_type, calc_score, InsideScoring, CountScoring,

# Chart parsing
Chart, chartparse

# include main content
include("core.jl")

# include submodules
include("ConjugateModels.jl")
include("BinaryCountGrammar.jl")
include("HeadedGrammars.jl")

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
