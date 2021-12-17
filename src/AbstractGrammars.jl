module AbstractGrammars

###############
### Exports ###
###############

export 
# utils
Tag, ⊣, @tag_str, normalize, default,

# Category interface
AbstractCategory, isstart, isnonterminal, isterminal,
StdCategory, start_category, terminal_category, nonterminal_category,

# Rule interface
AbstractRule, arity, apply, App,
StdRule, -->,

# Grammar interface
AbstractGrammar, push_completions!,
StdGrammar,

# Scorings
InsideScoring, CountScoring, BooleanScoring, BestDerivationScoring,
WDS, sample_derivations,

# Chart parsing
Chart, chartparse,

# Trees
Tree, labels, innerlabels, leaflabels, tree_similarity, isleaf, 
Treelet, treelets, 
dict2tree, tree2derivation, treelet2stdrule

###############
### Imports ###
###############

import Distributions: logpdf
import Base: zero, iszero, insert!, map, eltype, show

using LogProbs
using ShortStrings: ShortString31
using MLStyle: @match

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

# include submodules
include("AtMosts.jl")
# include("ConjugateModels.jl")

# main module code
include("main.jl")
include("chartparse.jl")
include("scorings.jl")
include("trees.jl")

end # module
