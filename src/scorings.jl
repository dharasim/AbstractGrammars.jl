mul_scores(scoring::Scoring, s1, s2, s3) = 
  mul_scores(scoring, s1, mul_scores(scoring, s2, s3))

######################
### Inside scoring ###
######################

struct InsideScoring{D} <: Scoring
  ruledist :: D
end

scoretype(::InsideScoring, grammar) = LogProb

function ruleapp_score(sc::InsideScoring, lhs, rule)
  LogProb(logpdf(sc.ruledist(lhs), rule), islog=true)
end

add_scores(::InsideScoring, left, right) = left + right
mul_scores(::InsideScoring, left, right) = left * right

# # test
# d = symdircat_ruledist(['a', 'b'], ['a' --> 'b', 'b' --> 'b'])
# sc = InsideScoring(d)
# @assert scoretype(sc, nothing) == LogProb
# @assert isone(ruleapp_score(sc, 'a', 'a' --> 'b'))
# @assert iszero(ruleapp_score(sc, 'a', 'a' --> 'a'))

#####################
### Count scoring ###
#####################

struct CountScoring <: Scoring end
scoretype(::CountScoring, grammar) = Int
ruleapp_score(::CountScoring, ::Any, ::Any) = 1
add_scores(::CountScoring, left, right) = left + right
mul_scores(::CountScoring, left, right) = left * right

#######################
### Boolean scoring ###
#######################

struct BooleanScoring <: Scoring end
scoretype(::BooleanScoring, grammar) = Bool
ruleapp_score(::BooleanScoring, ::Any, ::Any) = true
add_scores(::BooleanScoring, left, right) = left || right
mul_scores(::BooleanScoring, left, right) = left && right

###############################
### Best derivation scoring ###
###############################

struct BestDerivation{C, R<:Rule{C}}
  prob :: LogProb
  apps :: Vector{App{C, R}}
end

iszero(bd::BestDerivation) = iszero(bd.prob)

function zero(::Type{BestDerivation{C, R}}) where {C, R}
  BestDerivation(zero(LogProb), App{C, R}[])
end

struct BestDerivationScoring{D} <: Scoring
  ruledist :: D
end

function scoretype(::BestDerivationScoring, ::Grammar{R}) where {C, R<:Rule{C}} 
  BestDerivation{C, R}
end

function ruleapp_score(sc::BestDerivationScoring, lhs, rule)
  BestDerivation(
    LogProb(logpdf(sc.ruledist(lhs), rule), islog=true), 
    [App(lhs, rule)])
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

struct ScoredFreeEntry{S, T}
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
  ) where {S, T}
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
  ) where {S, T}
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

zero(::Type{ScoredFreeEntry{S, T}}) where {S,T} = ScoredFreeEntry(S, T)
iszero(x::ScoredFreeEntry) = x.tag == ZERO

# Weighted Derivation Scoring (WDS)
struct WDS{D, T, L} <: Scoring
  ruledist :: D
  store    :: Vector{ScoredFreeEntry{LogProb, T}}
  logpdf   :: L
end

function WDS(ruledist, ::Grammar{R}, logpdf=logpdf) where {C, R <: Rule{C}}
  WDS(ruledist, ScoredFreeEntry{LogProb, App{C, R}}[], logpdf)
end

function scoretype(::WDS, ::Grammar{R}) where {C, R<:Rule{C}} 
  ScoredFreeEntry{LogProb, App{C, R}}
end

function ruleapp_score(sc::WDS, lhs, rule)
  logp = LogProb(sc.logpdf(sc.ruledist(lhs), rule), islog=true)
  ScoredFreeEntry(sc.store, logp, App(lhs, rule))
end

function add_scores(sc::WDS, x, y)
  ZERO == x.tag && return y
  ZERO == y.tag && return x
  return ScoredFreeEntry(sc.store, +, x, y)
end

function mul_scores(sc::WDS, x, y)
  ZERO == x.tag && return x
  ZERO == y.tag && return y
  return ScoredFreeEntry(sc.store, *, x, y)
end

function sample_derivations(
    sc::WDS, x::ScoredFreeEntry{S, T}, n::Int
  ) where {S, T}
  vals = Vector{T}()
  for _ in 1:n
    sample_derivation!(vals, sc, x)
  end
  vals
end

function sample_derivation!(vals, sc::WDS, x::ScoredFreeEntry{S,T}) where {S, T}
  if VAL  ⊣ x 
    push!(vals, x.value)
  elseif ADD ⊣ x
    goleft = rand(S) < sc.store[x.leftIndex].score / x.score
    index = goleft ? x.leftIndex : x.rightIndex
    sample_derivation!(vals, sc, sc.store[index])
  elseif MUL ⊣ x
    sample_derivation!(vals, sc, sc.store[x.leftIndex])
    sample_derivation!(vals, sc, sc.store[x.rightIndex])
  else # ZERO ⊣ x
    error("cannot sample from zero")
  end
end

########################################################
### Free-semiring scorings with re-computable scores ###
########################################################

# To be done