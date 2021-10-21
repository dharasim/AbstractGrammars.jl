#########################
### Grammar Interface ###
#########################

abstract type AbstractRule{Category} end

abstract type AbstractGrammar{Rule<:AbstractRule} end

function push_completions!(grammar::G, stack, c) where G<:AbstractGrammar
  error("Function push_completions! (unary) not implemented for $G.")
end

function push_completions!(grammar::G, stack, c1, c2) where G<:AbstractGrammar
  error("Function push_completions! (binary) not implemented for $G.")
end

function logpdf(grammar::G, lhs, rule) where G<:AbstractGrammar
  error("Function logpdf not implemented for $G.")
end

# Rule applications
struct App{C, R <: AbstractRule{C}}
  lhs  :: C
  rule :: R
end

apply(::AbstractGrammar, rule, category) = apply(rule, category)
apply(grammar::AbstractGrammar, app::App) = apply(grammar, app.rule, app.lhs)

#########################
### Scoring interface ###
#########################

abstract type Scoring end

function score_type(grammar::AbstractGrammar, scoring::Scoring) 
  score_type(typeof(grammar), typeof(scoring))
end

function score_type(::Type{G}, ::Type{S}) where 
  {G <: AbstractGrammar, S <: Scoring}
  error("Function score_type not implemented for $G with $S.")
end

function ruleapp_score(scoring, grammar, lhs, rule)
  error("Function ruleapp_score not implemented")
end

function add_scores(scoring, left, right) 
  error("Function add_scores not implemented") 
end

function mul_scores(scoring, left, right)
  error("Function mul_scores not implemented") 
end

mul_scores(scoring::Scoring, s1, s2, s3) = 
  mul_scores(scoring, s1, mul_scores(scoring, s2, s3))