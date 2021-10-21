module AbstractGrammars

###############
### Exports ###
###############

export 
# utils
normalize, ⊣, default,

# Rule and grammar interface
AbstractRule, AbstractGrammar, App, apply, push_completions!,

# Scorings
InsideScoring, CountScoring, BooleanScoring, BestDerivationScoring,
WDS, sample_derivations,

# Chart parsing
Chart, chartparse,

# Trees
Tree, Binary, Leaf, dict2tree, innerlabels, leaflabels, tree_similarity

###############
### Imports ###
###############

import Distributions: logpdf
import Base: zero, iszero, insert!, map

using LogProbs

#############
### Utils ###
#############

# check for the tag of an object
⊣(tag, x) = x.tag == tag
⊣(tag, xs::Tuple) = all(x -> x.tag == tag, xs)

# default values for some types
default(::Type{T}) where T <: Number = zero(T)
default(::Type{Symbol}) = Symbol()

# generic normalization function
normalize(xs) = xs ./ sum(xs)

######################
### Included files ###
######################

# main module code
include("interface.jl")
include("chartparse.jl")
include("scorings.jl")
include("trees.jl")

# include submodules
include("ConjugateModels.jl")
include("Headed.jl")
include("HeadedSimple.jl")
include("AtMosts.jl")

end # module
