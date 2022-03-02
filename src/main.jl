# Type variables
# C ... category type
# R ... rule type
# A ... rule application type
# S ... score type

######################
### Abstract types ###
######################

abstract type Rule{C} end
abstract type Grammar{R<:Rule} end
abstract type Scoring end

#########################
### Rule applications ###
#########################

struct App{C, R<:Rule{C}}
  lhs  :: C
  rule :: R
end

show(io::IO, app::App) = print(io::IO, "App($(app.lhs), $(app.rule))")
apply(grammar, app::App) = apply(grammar, app.rule, app.lhs)

############################
### Type-level functions ###
############################

catytype(x) = categorytype(typeof(x))
catytype(R::Type{<:Rule{C}}) where {C} = C
catytype(G::Type{<:Grammar{R}}) where {R} = categorytype(R)
catytype(A::Type{App{C, R}}) where {C,R} = C

ruletype(x) = ruletype(typeof(x))
ruletype(R::Type{<:Rule}) = R
ruletype(G::Type{<:Grammar{R}}) where {R} = R
ruletype(A::Type{App{C, R}}) where {C,R} = R

apptype(x) = apptype(ruletype(x))
apptype(T::Type) = App{categorytype(T), ruletype(T)}

##########################
### Category interface ###
##########################

"""
    isnonterminal(category) ::Bool
"""
isnonterminal(category) = !isterminal(category)

"""
    isterminal(category) ::Bool
"""
isterminal(category) = !isnonterminal(category)

########################################
### Standard category implementation ###
########################################

struct StdCategory{T}
  isterminal :: Bool
  val        :: T
end

T(val) = StdCategory(true, val)
NT(val) = StdCategory(false, val)

T(c::StdCategory)  = @set c.isterminal = true
NT(c::StdCategory) = @set c.isterminal = false

isterminal(c::StdCategory) = c.isterminal
default(::Type{StdCategory{T}}) where T = StdCategory(default(Bool), default(T))

function show(io::IO, c::StdCategory)
  if isterminal(c)
    print(io, "T($(c.val))")
  else
    print(io, "NT($(c.val))")
  end
end

######################
### Rule interface ###
######################

"""
    arity(rule::Rule) ::Int

Rules have constant length of their right-hand sides and 
`arity(rule)` returns this length.
"""
function arity end

"""
    apply([grammar,] rule, category)

Apply `rule` to `category`, potentially using information from `grammar`.
Returns `nothing` if `rule` is not applicable to `category`.
Default implementation doesn't use `grammar`'s information and calls
`apply(rule, category)`.
"""
apply(_grammar, rule, category) = apply(rule, category)
apply(_rule::Rule, _category) = nothing

function apply(rule::Rule{C}, categories::Vector{C}) where C
  for (i,c) in enumerate(categories)
    if isnonterminal(c)
      d = apply(rule, c)
      if isnothing(d)
        return error("rule not applicable to leftmost nonterminal category")
      else
        return [categories[1:i-1]; collect(d); categories[i+1:end]]
      end
    end
  end
end

isapplicable(rule) = category -> isapplicable(rule, category)
isapplicable(rule, category) = !isnothing(apply(rule, category))
isapplicable(grammar, rule, category) = !isnothing(apply(grammar, rule, category))

####################################
### Standard rule implementation ###
####################################

struct StdRule{C} <: Rule{C}
  lhs :: C
  rhs :: AtMost{C, 2}
end

StdRule(lhs, rhs...) = StdRule(lhs, atmost2(rhs...))
-->(lhs::C, rhs::C) where C = StdRule(lhs, rhs)
-->(lhs::C, rhs) where C = StdRule(lhs, rhs...)

apply(r::StdRule{C}, c::C) where C = r.lhs == c ? tuple(r.rhs...) : nothing
arity(r::StdRule) = length(r.rhs)

function show(io::IO, r::StdRule)
  print(io, "$(r.lhs) -->")
  foreach(c -> print(io, " $c"), r.rhs)
end

macro rules(exprs...)
  # for example: evaluate :(a | b c | d) to [:a, :|, :b, :c, :|, :d]
  function flatten_or_expression(ex)
    if ex isa Expr && ex.head == :call && ex.args[1] == :|
      l = flatten_or_expression(ex.args[2])
      r = ex.args[3]
      [l..., :|, r]
    else
      [ex]
    end
  end

  function flatten_or_expressions(exprs...)
    vcat(map(flatten_or_expression, exprs)...)
  end

  function split_by_pipe(v::Vector)
    a = [v; :(|)]
    splits = [0; findall(isequal(:|), a)]
    splitstarts, splitends = @view(splits[1:end-1]), @view(splits[2:end])
    [view(a, i1+1:i2-1) for (i1, i2) in zip(splitstarts, splitends)]
  end

  @assert exprs[1].head == :(-->) "Rules must be declared with a long arrow -->"
  lhs = exprs[1].args[1]
  rhss = split_by_pipe(flatten_or_expressions(exprs[1].args[2], exprs[2:end]...))
  rs = [StdRule(lhs, rhs...) for rhs in rhss]
  :($rs)
end

#########################
### Grammar interface ###
#########################

"""
    push_completions!(grammar, stack, c1[, c2])

Push all unary completions of `c1` or all binary completions of `(c1, c2)` 
on `stack`. Completions are typed as rule applications.
"""
function push_completions! end

function observe_app!(ruledist, app, k=1)
  if insupport(ruledist(app.lhs), app.rule)
    add_obs!(ruledist(app.lhs), app.rule, k)
  else
    @info "Rule $(app.rule) not observed. It's not in the rule distribution."
  end
end

function observe_tree!(treelet2rule, ruledist, tree)
  for app in tree2apps(treelet2rule, tree)
    observe_app!(ruledist, app)
  end
end

function observe_trees!(treelet2rule, ruledist, trees)
  for tree in trees
    observe_tree!(treelet2rule, ruledist, tree)
  end
end

#######################################
### Standard grammar implementation ###
#######################################

mutable struct StdGrammar{C} <: Grammar{StdRule{C}}
  start       :: Set{C}
  rules       :: Set{StdRule{C}}
  completions :: Dict{AtMost{C, 2}, Vector{C}}

  function StdGrammar(start, rules)
    R = eltype(typeof(rules))
    C = catytype(R)
    completions = Dict{AtMost{C, 2}, Vector{C}}()
    for r in rules
      comps = get!(() -> C[], completions, r.rhs)
      push!(comps, r.lhs)
    end
    return new{C}(Set(collect(start)), Set(collect(rules)), completions)
  end
end

function push_completions!(grammar::StdGrammar, stack, categories...)
  rhs = atmost2(categories...)
  if haskey(grammar.completions, rhs)
    for lhs in grammar.completions[rhs]
      push!(stack, App(lhs, StdRule(lhs, rhs...)))
    end
  end
end

##########################
### Rule distributions ###
##########################

struct DirCatRuleDist{C, R<:Rule{C}}
  dists :: Dict{C, DirCat{R}}
end

(d::DirCatRuleDist)(c) = d.dists[c]

function symdircat_ruledist(categories, rules, concentration=1.0)
  applicable_rules(c) = filter(r -> isapplicable(r, c), rules)
  dists = Dict(
    c => symdircat(applicable_rules(c), concentration) 
    for c in categories
  )
  DirCatRuleDist(dists)
end

struct ConstDirCatRuleDist{R<:Rule}
  dist :: DirCat{R}
end

(d::ConstDirCatRuleDist)(c) = d.dist

#####################
### Product rules ###
#####################

struct ProductRule{C1, C2, R1<:Rule{C1}, R2<:Rule{C2}} <: Rule{Tuple{C1, C2}}
  rule1 :: R1
  rule2 :: R2

  function ProductRule(rule1::R1, rule2::R2) where {
      C1, C2, R1 <: Rule{C1}, R2 <: Rule{C2}
    }
    @assert arity(rule1) == arity(rule2)
    new{C1, C2, R1, R2}(rule1, rule2)
  end
end

import Base: getindex
function getindex(rule::ProductRule, i)
  if i == 1
    rule.rule1
  elseif i == 2
    rule.rule2
  else
    BoundsError(rule, i)
  end
end

show(io::IO, r::ProductRule) = print(io, "($(r[1]), $(r[2]))")
arity(rule::ProductRule) = arity(rule[1])

function apply(rule::ProductRule{C1,C2}, category::Tuple{C1,C2}) where {C1,C2}
  rhs1 = apply(rule[1], category[1])
  rhs2 = apply(rule[2], category[2])
  if isnothing(rhs1) || isnothing(rhs2)
    nothing
  else
    tuple(zip(rhs1, rhs2)...)
  end
end

########################
### Product Grammars ###
########################

# not thread safe
# for parallelization use one product grammar per thread
mutable struct ProductGrammar{
    C1, R1<:Rule{C1}, G1<:Grammar{R1}, 
    C2, R2<:Rule{C2}, G2<:Grammar{R2},
  } <: Grammar{ProductRule{C1,C2,R1,R2}}

  grammar1 :: G1
  grammar2 :: G2
  stacks   :: Tuple{Vector{App{C1, R1}}, Vector{App{C2, R2}}}

  function ProductGrammar(grammar1::G1, grammar2::G2) where {
      C1, R1<:Rule{C1}, G1<:Grammar{R1}, 
      C2, R2<:Rule{C2}, G2<:Grammar{R2},
    }
    stacks = tuple(Vector{App{C1, R1}}(), Vector{App{C2, R2}}())
    new{C1,R1,G1,C2,R2,G2}(grammar1, grammar2, stacks)
  end
end

function getindex(grammar::ProductGrammar, i)
  if i == 1
    grammar.grammar1
  elseif i == 2
    grammar.grammar2
  else
    BoundsError(grammar, i)
  end
end

function push_completions!(grammar::ProductGrammar, stack, categories...)
  function unzip(xs)
    n = length(first(xs))
    ntuple(i -> map(x -> x[i], xs), n)
  end

  rhss = unzip(categories) # right-hand sides
  push_completions!(grammar[1], grammar.stacks[1], rhss[1]...)
  push_completions!(grammar[2], grammar.stacks[2], rhss[2]...)

  for app1 in grammar.stacks[1], app2 in grammar.stacks[2]
    app = App((app1.lhs, app2.lhs), ProductRule(app1.rule, app2.rule))
    push!(stack, app)
  end

  empty!(grammar.stacks[1])
  empty!(grammar.stacks[2])
  return nothing
end

#############################
### Variational inference ###
#############################

function estimate_rule_counts(
    ruledist, grammar, sequences, seq2start, seq2numtrees=seq->length(seq)^2;
    showprogress=true
  )
  function single_estimate(sequence)
    scoring = WDS(ruledist, grammar, logvarpdf)
    chart = chartparse(grammar, scoring, sequence)
    forest = chart[1, end][seq2start(sequence)]
    n = seq2numtrees(sequence)
    return 1/n * counter(sample_derivations(scoring, forest, n))
  end
  #
  p = Progress(
    length(sequences); 
    desc="estimating rule counts: ", 
    enabled=showprogress
  )
  estimates_per_sequence = progress_map(single_estimate, sequences; progress=p)
  reduce(merge!, estimates_per_sequence)
end

# run variational inference
function runvi(epochs, mk_prior, estimation_args...; showprogress=true)
  ruledist = first(estimation_args) # bring into scope
  for e in 1:epochs
    showprogress ? println("epoch $e of $epochs") : nothing
    rule_counts = estimate_rule_counts(estimation_args...; showprogress)
    ruledist = mk_prior()
    for (app, pscount) in rule_counts
      add_obs!(ruledist(app.lhs), app.rule, pscount)
    end
  end
  ruledist
end

# run variational inference with automatic prior initialization
function runvi(
    epochs, mk_prior, grammar::Grammar, other_estimation_args...;
    showprogress=true
  )
  runvi(epochs, mk_prior, mk_prior(), grammar, other_estimation_args...; showprogress)
end