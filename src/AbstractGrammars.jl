module AbstractGrammars

###############
### Exports ###
###############

export 
# utils
Tag, ⊣, @tag_str, normalize, default,

# Category interface
isnonterminal, isterminal, 
StdCategory, NT, T,

# Rule interface
Rule, arity, apply, App,
StdRule, -->, @rules,
ProductRule,

# Grammar interface
Grammar, push_completions!,
StdGrammar, ProductGrammar,

# Variational inference
estimate_rule_counts, runvi,

# Scorings
InsideScoring, CountScoring, BooleanScoring, 
BestDerivationScoring, getbestderivation,
AllDerivationScoring, getallderivations,
WDS, sample_derivations,

# Rule distributions
DirCatRuleDist, symdircat_ruledist, ConstDirCatRuleDist,
observe_app!, observe_tree!, observe_trees!,

# Chart parsing
Chart, chartparse,

# Trees
Tree, labels, innerlabels, leaflabels, tree_similarity, isleaf, zip_trees,
Treelet, treelets, 
dict2tree, tree2derivation, tree2apps, treelet2stdrule,
plot_tree

###############
### Imports ###
###############

import Distributions: logpdf, insupport
import Base: zero, iszero, insert!, map, eltype, show, +, *

using SimpleProbabilisticPrograms: BetaBinomial, add_obs!, DirCat, symdircat, logvarpdf
using LogProbs
using ShortStrings: ShortString31
using Setfield: @set
using ProgressMeter: Progress, progress_map
using DataStructures: counter, Accumulator

#############
### Utils ###
#############

*(a::Accumulator, n::Number) = Accumulator(Dict(k => v*n for (k,v) in a.map))
*(n::Number, a::Accumulator) = Accumulator(Dict(k => n*v for (k,v) in a.map))
+(a::Accumulator, n::Number) = Accumulator(Dict(k => v+n for (k,v) in a.map))
+(n::Number, a::Accumulator) = Accumulator(Dict(k => n+v for (k,v) in a.map))

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
default(::Type{QuoteNode}) = QuoteNode(Symbol())
default(::Type{String}) = ""
default(::Type{Char}) = ' '
default(::Type{T}) where T <: AbstractVector = T()

function default(::Type{T}) where T
  T(map(default, fieldtypes(T))...)
end

# generic normalization function
normalize(xs) = xs ./ sum(xs)

######################
### Included files ###
######################

include("PlotTree.jl") # submodule
using .PlotTree: plot_tree

include("AtMosts.jl") # submodule
using .AtMosts: AtMost, atmost2

include("main.jl")
include("chartparse.jl")
include("scorings.jl")
include("trees.jl")

include("JazzTreebank.jl") # submodule

end # module
