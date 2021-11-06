module AbstractGrammars

###############
### Exports ###
###############

export 
# utils
Tag, ⊣, @tag_str, normalize, default,

# Rule and grammar interface
AbstractRule, AbstractGrammar, App, apply, arity, push_completions!,

# Scorings
InsideScoring, CountScoring, BooleanScoring, BestDerivationScoring,
WDS, sample_derivations,

# Chart parsing
Chart, chartparse,

# Trees
Tree, labels, innerlabels, leaflabels, tree_similarity, isleaf, tree2derivation,
Treelet, treelets

###############
### Imports ###
###############

import Distributions: logpdf
import Base: zero, iszero, insert!, map, eltype

using LogProbs
using ShortStrings: ShortString31

#############
### Utils ###
#############

# check for the tag of an object
⊣(tag, x) = x.tag == tag
⊣(tag, xs::Tuple) = all(x -> x.tag == tag, xs)

# construct tags as short strings
macro tag_str(str)
  ShortString31(str)
end

const Tag = ShortString31

# default values for some types
default(::Type{T}) where T <: Number = zero(T)
default(::Type{Symbol}) = Symbol()
default(::Type{Char}) = ' '

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
include("AtMosts.jl")
include("PCFG.jl")
include("ConjugateModels.jl")
include("GeneralCategories.jl")
include("Headed.jl")
include("HeadedTyped.jl")

end # module
