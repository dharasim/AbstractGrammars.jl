######################
### Inside scoring ###
######################

struct InsideScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{InsideScoring}) = LogProb
ruleapp_score(::InsideScoring, grammar::AbstractGrammar, lhs, rule) =
  LogProb(logpdf(grammar, lhs, rule), islog=true)
add_scores(::InsideScoring, left, right) = left + right
mul_scores(::InsideScoring, left, right) = left * right

#####################
### Count scoring ###
#####################

struct CountScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{CountScoring}) = Int
ruleapp_score(::CountScoring, ::AbstractGrammar, ::Any, ::Any) = 1
add_scores(::CountScoring, left, right) = left + right
mul_scores(::CountScoring, left, right) = left * right

#######################
### Boolean scoring ###
#######################

struct BooleanScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{BooleanScoring}) = Bool
ruleapp_score( ::BooleanScoring, ::AbstractGrammar, ::Any, ::Any) = true
add_scores(::BooleanScoring, left, right) = left || right
mul_scores(::BooleanScoring, left, right) = left && right

###############################
### Best derivation scoring ###
###############################

struct BestDerivation{C, R<:AbstractRule{C}}
  prob :: LogProb
  apps :: Vector{App{C, R}}
end

iszero(bd::BestDerivation) = iszero(bd.prob)

function zero(::Type{BestDerivation{C, R}}) where {C, R}
  BestDerivation(zero(LogProb), App{C, R}[])
end

struct BestDerivationScoring <: Scoring end

function score_type(::Type{G}, ::Type{BestDerivationScoring}) where 
    {C, R <: AbstractRule{C}, G <: AbstractGrammar{R}}
  BestDerivation{C, R}
end

function ruleapp_score(::BestDerivationScoring, grammar::AbstractGrammar, lhs, rule)
  BestDerivation(LogProb(logpdf(grammar, lhs, rule), islog=true), [App(lhs, rule)])
end

function add_scores(::BestDerivationScoring, left, right)
  left.prob >= right.prob ? left : right
end

function mul_scores(::BestDerivationScoring, left, right)
  BestDerivation(left.prob * right.prob, [left.apps; right.apps])
end

######################################################################
### Free-semiring scorings with manually managed pointer structure ###
######################################################################

# Implementation idea: break rec. structure with indices into a vector (store).
# Ihe store contains unboxed values, which reduces GC times.
# Additionally, it allows to update probabilities without parsing again 
# (not yet implemented).

@enum ScoreTag ADD MUL VAL ZERO

struct ScoredFreeEntry{S,T}
  tag        :: ScoreTag
  score      :: S
  value      :: T
  index      :: Int
  leftIndex  :: Int
  rightIndex :: Int

  # addition and multiplication
  function ScoredFreeEntry(
    store :: Vector{ScoredFreeEntry{S,T}},
    op    :: Union{typeof(+), typeof(*)},
    left  :: ScoredFreeEntry{S,T}, 
    right :: ScoredFreeEntry{S,T}
  ) where {S,T}
    tag(::typeof(+)) = ADD
    tag(::typeof(*)) = MUL
    score = op(left.score, right.score)
    value = left.value # dummy value
    index = length(store) + 1
    x = new{S,T}(tag(op), score, value, index, left.index, right.index)
    push!(store, x)
    return x
  end

  # scored values
  function ScoredFreeEntry(
    store :: Vector{ScoredFreeEntry{S,T}},
    score :: S,
    value :: T
  ) where {S,T}
    index = length(store) + 1
    x = new{S,T}(VAL, score, value, index)
    push!(store, x)
    return x
  end

  # constant zero
  function ScoredFreeEntry(::Type{S}, ::Type{T}) where {S,T}
    new{S,T}(ZERO, zero(S))
  end
end

zero(::Type{ScoredFreeEntry{S,T}}) where {S,T} = ScoredFreeEntry(S, T)
iszero(x::ScoredFreeEntry) = x.tag == ZERO

# Weighted Derivation Scoring (WDS)
struct WDS{S,T} <: Scoring
  store :: Vector{ScoredFreeEntry{S,T}}
end

function WDS(::G) where
  {C, R <: AbstractRule{C}, G <: AbstractGrammar{R}}
  WDS(ScoredFreeEntry{LogProb, App{C, R}}[])
end

score_type(::Type{<:AbstractGrammar}, ::Type{<:WDS{S,T}}) where {S, T} =
  ScoredFreeEntry{S, T}
ruleapp_score(s::WDS, grammar, lhs, rule) = 
  ScoredFreeEntry(
    s.store, 
    LogProb(logpdf(grammar, lhs, rule), islog=true), 
    App(lhs, rule))

function add_scores(s::WDS, x, y)
  ZERO == x.tag && return y
  ZERO == y.tag && return x
  return ScoredFreeEntry(s.store, +, x, y)
end

function mul_scores(s::WDS, x, y)
  ZERO == x.tag && return x
  ZERO == y.tag && return y
  return ScoredFreeEntry(s.store, *, x, y)
end

function sample_derivations(
    s::WDS{S,T}, x::ScoredFreeEntry{S,T}, n::Int
  ) where {S,T}
  
  vals = Vector{T}()
  for _ in 1:n
    sample_derivation!(vals, s, x)
  end
  vals
end

function sample_derivation!(vals, s::WDS, x::ScoredFreeEntry{S,T}) where {S,T}
  if VAL  ⊣ x 
    push!(vals, x.value)
  elseif ADD ⊣ x
    index = rand(S) < s.store[x.leftIndex].score / x.score ? x.leftIndex : x.rightIndex
    sample_derivation!(vals, s, s.store[index])
  elseif MUL ⊣ x
    sample_derivation!(vals, s, s.store[x.leftIndex])
    sample_derivation!(vals, s, s.store[x.rightIndex])
  else # ZERO ⊣ x
    error("cannot sample from zero")
  end
end

########################################################
### Free-semiring scorings with re-computable scores ###
########################################################

# To be done