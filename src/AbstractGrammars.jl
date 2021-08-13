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

################
### Includes ###
################

include("interfaces.jl")
include("scorings.jl")
include("chartparsing.jl")
include("ConjugateModels.jl")

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
