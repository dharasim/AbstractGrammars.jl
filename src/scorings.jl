######################
### Inside scoring ###
######################

struct InsideScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{InsideScoring}) = LogProb

function calc_score(grammar::AbstractGrammar, ::InsideScoring, lhs, rule)
  LogProb(logpdf(grammar.ruledist, lhs, rule))
end

#####################
### Count scoring ###
#####################

struct CountScoring <: Scoring end
score_type(::Type{<:AbstractGrammar}, ::Type{CountScoring}) = Int

function calc_score(grammar::AbstractGrammar, ::CountScoring, lhs, rule)
  1
end