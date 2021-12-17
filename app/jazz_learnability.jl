using AbstractGrammars

# imports for overloading
import AbstractGrammars: default
import Distributions: logpdf, BetaBinomial

# imports without overloading
# using AbstractGrammars.ConjugateModels: DirCat, add_obs!
using Statistics: mean
using SimpleProbabilisticPrograms
using Pitches: parsespelledpitch, Pitch, SpelledIC, MidiIC, midipc, alteration, @p_str, tomidi
using Underscores: @_
using MLStyle: @match
using ProgressMeter: @showprogress

# named imports
# import AbstractGrammars.Headed
import HTTP, JSON

#############
### Utils ###
#############

default(::Type{SpelledIC}) = SpelledIC(0)
default(::Type{Pitch{I}}) where I = Pitch(default(I))

##############
### Chords ###
##############

import Base: show

@enum ChordForm MAJ MAJ6 MAJ7 DOM MIN MIN6 MIN7 MINMAJ7 HDIM7 DIM7 SUS

const chordform_strings = 
  ["^", "6", "^7", "7", "m", "m6", "m7", "m^7", "%7", "o7", "sus"]

chordform_string(form::ChordForm) = chordform_strings[Int(form) + 1]

function parse_chordform(str::AbstractString)
  i = findfirst(isequal(str), chordform_strings)
  @assert !isnothing(i) "$str cannot be parsed as a chord form"
  return ChordForm(i-1)
end

default(::Type{ChordForm}) = ChordForm(0)

@assert all(instances(ChordForm)) do form
  form |> chordform_string |> parse_chordform == form
end

struct Chord{R}
  root :: R
  form :: ChordForm
end

show(io::IO, chord::Chord) = print(io, chord.root, chordform_string(chord.form))

function default(::Type{Chord{R}}) where R 
  Chord(default(R), default(ChordForm))
end

default(Chord{Pitch{SpelledIC}})

const chord_regex = r"([A-G]b*|[A-G]#*)([^A-Gb#]+)" 

function parse_chord(str)
  m = match(chord_regex, str)
  @assert !isnothing(m) "$str cannot be parsed as a pitch-class chord"
  root_str, form_str = m.captures
  root = parsespelledpitch(root_str)
  # pitchclass_root = tomidi(spelled_root)
  form = parse_chordform(form_str)
  return Chord(root, form)
end

#####################
### Read treebank ###
#####################

# Tonal Pitch-Class Chord
const TPCC = Chord{Pitch{SpelledIC}}

function categorize_and_insert_unary_rules(tree; root=start_category(TPCC))
  function categorize(tree)
    if isleaf(tree)
      Tree(nonterminal_category(tree.label),Tree(terminal_category(tree.label)))
    else
      Tree(nonterminal_category(tree.label), map(categorize, tree.children))
    end
  end
  Tree(root, categorize(tree))
end

function preprocess_tree!(tune)
  remove_asterisk(label::String) = replace(label, "*" => "")

  if haskey(tune, "trees")
    tune["tree"] = @_ tune["trees"][1]["open_constituent_tree"] |> 
      dict2tree(remove_asterisk, __) |>
      map(parse_chord, __) |>
      categorize_and_insert_unary_rules(__)
  end

  return tune
end

treebank_url = "https://raw.githubusercontent.com/DCMLab/JazzHarmonyTreebank/master/treebank.json"
tunes = HTTP.get(treebank_url).body |> String |> JSON.parse .|> preprocess_tree!
treebank = filter(tune -> haskey(tune, "tree"), tunes)

#########################
### Construct grammar ###
#########################

all_chords = collect(
  Chord(parsespelledpitch(letter * acc), form) 
  for letter in 'A':'G'
  for acc in ("b", "#", "")
  for form in instances(ChordForm))

START = start_category(TPCC)
ts    = terminal_category.(all_chords)
nts   = nonterminal_category.(all_chords) 

rules = Set([
  [START --> nt for nt in nts]; # start rules
  [nt  --> t         for (nt, t) in zip(nts, ts)]; # termination rules
  [nt  --> (nt,nt)   for nt in nts]; # duplication rules
  # [nt1 --> (nt1,nt2) for nt1 in nts for nt2 in nts if nt1 != nt2]; #left-headed
  [nt2 --> (nt1,nt2) for nt1 in nts for nt2 in nts if nt1 != nt2]; #right-headed
  ])

# probability model
applicable_rules(all_rules, category) = filter(r -> r.lhs==category, all_rules)
prior_params() = Dict(
  nt => flat_dircat(applicable_rules(rules, nt)) for nt in [nts; START])

function logpdf(grammar::StdGrammar, lhs, rule)
  if lhs == rule.lhs
    logpdf(grammar.params[lhs], rule)
  else
    log(0)
  end
end

# supervised training by observation of trees
function observe_tree!(params, tree)
  for rule in tree2derivation(treelet2stdrule, tree)
    try
      add_obs!(params[rule.lhs], rule, 1)
    catch
      print("x")
    end
  end
end

grammar = StdGrammar([START], rules, prior_params())
foreach(tune -> observe_tree!(grammar.params, tune["tree"]), treebank)

############################
### Test with dummy data ###
############################

# terminalss = collect([H.terminal_cat(c)]
#   for c in [Chord(p"C", MAJ7), Chord(p"G", DOM), Chord(p"C", MAJ7)])
# terminalss = fill([terminal_category(Chord(p"C", MAJ7))], 50)
terminalss = [[terminal_category(rand(all_chords))] for _ in 1:50]
scoring = WDS(grammar) # weighted derivation scoring
@time chart = chartparse(grammar, scoring, terminalss)
@time sample_derivations(scoring, chart[1,length(terminalss)][START], 1) .|> 
  (app -> arity(app.rule))

##########################
### Test with treebank ###
##########################

function calc_accs(grammar, treebank, startsymbol; treekey="tree")
  scoring = BestDerivationScoring()
  accs = zeros(length(treebank))
  for i in eachindex(treebank)
    tree = treebank[i][treekey]
    terminalss = [[c] for c in leaflabels(tree)]
    chart = chartparse(grammar, scoring, terminalss)
    apps = chart[1, length(terminalss)][startsymbol].apps
    derivation = [app.rule for app in apps]
    accs[i] = tree_similarity(tree, apply(derivation, startsymbol))
    # println(i, ' ', treebank[i]["title"], ' ', accs[i])
  end
  return accs
end

# @time accs = calc_accs(grammar, treebank[1:150], START)
# sum(accs) / length(accs)

########################
### Read rhythm data ###
########################

function chord_durations(tune)
  bpm = tune["meter"]["numerator"] # beats per measure
  ms  = tune["measures"]
  bs  = tune["beats"]
  n   = length(tune["chords"])

  @assert n == length(ms) == length(bs) "error in treebank's rhythm data"
  ds = zeros(Int, n) # initialize list of durations
  for i in 1:n-1
    b1, b2 = bs[i:i+1] # current and next beat
    m1, m2 = ms[i:i+1] # measure of current and next beat
    # The chord on the current beat offsets either at the next chord or
    # the end of the current measure.
    # This is by convention of the treebank annotations.
    ds[i] = m1 == m2 ? b2 - b1 : bpm + 1 - b1
  end
  ds[n] = bpm + 1 - bs[n]

  @assert all(d -> 0 < d, ds) "bug in chord-duration calculation or data"
  return ds
end

function leaf_durations(tune)
  ds = chord_durations(tune)
  ls = leaflabels(tune["tree"])
  if length(ds) == length(ls)
    ds
  elseif length(ds) + 1 == length(ls) # tune ends on its first chord
    [ds; sum(ds)]
  elseif length(ds) > length(ls) # turnaround is omitted in the tree
    [ds[1:length(ls)-1]; sum(ds[length(ls):end])]
  else # much more chords than chord durations
    error("list of chord durations not long enough")
  end
end

function normalized_duration_tree(tune)
  lds = normalize(Rational.(leaf_durations(tune)))
  k = 0 # leaf index
  next_leafduration() = (k += 1; lds[k])

  function relabel(tree) 
    if isleaf(tree)
      Tree(next_leafduration())
    elseif length(tree.children) == 1
      child = relabel(tree.children[1])
      Tree(child.label, child)
    elseif length(tree.children) == 2
      left  = relabel(tree.children[1])
      right = relabel(tree.children[2])
      Tree(left.label + right.label, left, right)
    else
      error("tree is not (even weakly) binary")
    end
  end

  return relabel(dict2tree(tune["trees"][1]["open_constituent_tree"]))
end

for tune in tunes
  if haskey(tune, "tree")
    tune["rhythm_tree"] = categorize_and_insert_unary_rules(
      normalized_duration_tree(tune), 
      root=start_category(Rational{Int}))
  end
end

# @time chord_durations(tune)
# @time leaf_durations(tune)
# @time normalized_duration_tree.(treebank)
# @time chord_durations.(tunes);

# failed = 0
# for tune in tunes
#   try
#     chord_durations(tune)
#   catch
#     failed += 1
#     println(tune["title"])
#   end
# end
# failed

######################
### Rhythm Grammar ###
######################

import AbstractGrammars: arity, apply, push_completions!

const RhythmCategory = StdCategory{Rational{Int}}

# possible tags: start, termination, split
struct RhythmRule <: AbstractRule{RhythmCategory}
  tag   :: Tag
  ratio :: Rational{Int}
end

function show(io::IO, r::RhythmRule)
  @match r.tag begin
    "start"       => print(io, "RhymStart()")
    "termination" => print(io, "RhymTerm()")
    "split"       => print(io, "RhymSplit($(r.ratio))")
  end
end

const rhythm_start_category = start_category(Rational{Int})
const rhythm_start_rule = RhythmRule("start", default(Rational{Int}))
const rhythm_termination = RhythmRule("termination", default(Rational{Int}))

rhythm_split_rule(ratio) = RhythmRule("split", ratio)
arity(rule::RhythmRule) = "split" ⊣ rule ? 2 : 1

function apply(rule::RhythmRule, category::RhythmCategory)
  if "start" ⊣ rule && "start" ⊣ category
    tuple(nonterminal_category(1//1))
  elseif "termination" ⊣ rule && "nonterminal" ⊣ category
    tuple(terminal_category(category))
  elseif "split" ⊣ rule && "nonterminal" ⊣ category
    tuple(
      nonterminal_category(rule.ratio * category.val), 
      nonterminal_category((1 - rule.ratio) * category.val))
  else
    nothing
  end
end

mutable struct RhythmGrammar{P} <: AbstractGrammar{RhythmRule}
  rules  :: Set{RhythmRule}
  params :: P

  function RhythmGrammar(rules, params::P) where P
    @assert rhythm_start_rule in rules && rhythm_termination in rules
    new{P}(rules, params)
  end
end

function push_completions!(::RhythmGrammar, stack, category)
  if "terminal" ⊣ category
    push!(stack, App(nonterminal_category(category), rhythm_termination))
  elseif "nonterminal" ⊣ category
    push!(stack, App(rhythm_start_category, rhythm_start_rule))
  end
end

function push_completions!(grammar::RhythmGrammar, stack, c1, c2)
  if "nonterminal" ⊣ c1 && "nonterminal" ⊣ c2
    s = sum(c1.val + c2.val)
    ratio = c1.val / s
    rule = rhythm_split_rule(ratio)
    if rule in grammar.rules
      push!(stack, App(nonterminal_category(s), rule))
    end
  end
end

function logpdf(grammar::RhythmGrammar, lhs, rule)
  if "start" ⊣ lhs && "start" ⊣ rule
    log(1)
  elseif "nonterminal" ⊣ lhs && rule in grammar.rules
    logpdf(grammar.params, rule)
  else # not applicable
    log(0)
  end
end

split_rules = Set([rhythm_split_rule(d//n) for d in 1:100 for n in d+1:100])
rhythm_rules = union(split_rules, [rhythm_start_rule, rhythm_termination])
params = flat_dircat([rhythm_termination; collect(split_rules)])
rhythm_grammar = RhythmGrammar(rhythm_rules, params)

tune = treebank[30]
terminalss = [[terminal_category(d)] for d in normalize(Rational.(chord_durations(tune)))]
scoring = WDS(rhythm_grammar)
@time chart = chartparse(rhythm_grammar, scoring, terminalss)
chart[1,length(terminalss)][rhythm_start_category]

# @time accs = calc_accs(rhythm_grammar, treebank, rhythm_start_category, treekey="rhythm_tree")
# sum(accs) / length(accs)

function treelet2rhythmrule(treelet)
  root = treelet.root_label
  children = treelet.child_labels
  if arity(treelet) == 1
    child = children[1]
    if "start" ⊣ root && nonterminal_category(1//1) == child
      rhythm_start_rule
    elseif "nonterminal" ⊣ root && "terminal" ⊣ child && root.val == child.val
      rhythm_termination
    else
      error("cannot convert unary $treelet into a rhythm rule")
    end
  elseif arity(treelet) == 2 && "nonterminal" ⊣ (root, children...) &&
         root.val == children[1].val + children[2].val
    rhythm_split_rule(children[1].val // root.val)
  else
    error("cannot convert binary $treelet into a rhythm rule")
  end
end

function observe_rhythm_tree!(params, tree)
  for rule in tree2derivation(treelet2rhythmrule, tree)
    if !("start" ⊣ rule)
      try
        add_obs!(params, rule, 1)
      catch
        print("x")
      end
    end
  end
end

for tune in treebank 
  observe_rhythm_tree!(params, tune["rhythm_tree"])
end

# @time accs = calc_accs(rhythm_grammar, treebank, rhythm_start_category, treekey="rhythm_tree")
# sum(accs) / length(accs)

#######################
### Product Grammar ###
#######################

struct ProductRule{C1, C2, R1 <: AbstractRule{C1}, R2 <: AbstractRule{C2}} <:  
    AbstractRule{Tuple{C1, C2}}
  rule1 :: R1
  rule2 :: R2

  function ProductRule(rule1::R1, rule2::R2) where 
      {C1, C2, R1 <: AbstractRule{C1}, R2 <: AbstractRule{C2}}
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

rule = ProductRule(rand(rules), rand(split_rules))
arity(rule)
c = (rule[1].lhs, nonterminal_category(1//1))
rhs = apply(rule, c)
@assert rhs[1] isa Tuple && typeof(rhs[1]) == typeof(rhs[2])

# not thread safe
# for parallelization use one product grammar per thread
mutable struct ProductGrammar{
    C1, R1<:AbstractRule{C1}, G1<:AbstractGrammar{R1}, 
    C2, R2<:AbstractRule{C2}, G2<:AbstractGrammar{R2},
    P
  } <: AbstractGrammar{ProductRule{C1,C2,R1,R2}}
  grammar1 :: G1
  grammar2 :: G2
  stacks   :: Tuple{Vector{App{C1, R1}}, Vector{App{C2, R2}}}
  params   :: P

  function ProductGrammar(grammar1::G1, grammar2::G2, params::P) where {
      C1, R1<:AbstractRule{C1}, G1<:AbstractGrammar{R1}, 
      C2, R2<:AbstractRule{C2}, G2<:AbstractGrammar{R2},
      P
    }
    stacks = tuple(Vector{App{C1, R1}}(), Vector{App{C2, R2}}())
    new{C1,R1,G1,C2,R2,G2,P}(grammar1, grammar2, stacks, params)
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

function logpdf(
    grammar::ProductGrammar{C1, R1, G1, C2, R2, G2, <:BetaBinomial}, lhs, rule
  ) where {C1, R1, G1, C2, R2, G2}

  beta_bernoulli = grammar.params
  @assert 1 <= arity(rule) <= 2
  arity_logprob = logpdf(beta_bernoulli, arity(rule)-1)
  *(
    arity_logprob,
    logpdf(grammar[1], lhs[1], rule[1]),
    logpdf(grammar[2], lhs[2], rule[2]))
end

function zip_trees(t1, t2)
  @assert length(t1.children) == length(t2.children)
  if isleaf(t1)
    Tree((t1.label, t2.label))
  else
    zipped_children = map(zip_trees, t1.children, t2.children)
    Tree((t1.label, t2.label), zipped_children)
  end
end 

for tune in treebank
  tune["product_tree"] = zip_trees(tune["tree"], tune["rhythm_tree"])
end

product_sequence(tune) = leaflabels(tune["product_tree"])

using DataStructures: counter
using DataStructures: Accumulator
import Base: *, +
*(a::Accumulator, n::Number) = Accumulator(Dict(k => v*n for (k,v) in a.map))
*(n::Number, a::Accumulator) = Accumulator(Dict(k => n*v for (k,v) in a.map))
+(a::Accumulator, n::Number) = Accumulator(Dict(k => v+n for (k,v) in a.map))
+(n::Number, a::Accumulator) = Accumulator(Dict(k => n+v for (k,v) in a.map))

function most_frequent(a::Accumulator, n=10)
  @_ collect(a.map) |>
     sort!(__, by=kv->kv[2], rev=true) |>
     first(__, n)
end

function estimate_rule_counts_single_sequence(
    grammar, sequence, startsym; num_trees=length(sequence)^2
  )
  scoring = WDS(grammar, logvarpdf)
  chart = chartparse(grammar, scoring, sequence)
  forest = chart[1,length(sequence)][startsym]
  return 1/num_trees * counter(sample_derivations(scoring, forest, num_trees))
end

function estimate_rule_counts(grammar, sequences, args...; kwargs...)
  estimates_per_sequnece = map(sequences) do seq
    estimate_rule_counts_single_sequence(grammar, seq, args...; kwargs...)
  end
  reduce(merge!, estimates_per_sequnece)
end

function variational_inference_step!(grammar, sequences, startsym, mk_prior)
  rule_counts = estimate_rule_counts(grammar, sequences, startsym)
  grammar.params = mk_prior()
  @showprogress for (app, pscount) in rule_counts
    add_obs!(rule_dist(grammar, app.lhs), app.rule, pscount)
  end
end

using Random: AbstractRNG
using Distributions: Beta, Geometric
using LogExpFunctions: logaddexp
using SpecialFunctions: logbeta, digamma
import Base: rand
import Distributions: logpdf
import SimpleProbabilisticPrograms: logvarpdf, add_obs!

mutable struct BetaGeometric
  α :: Float64
  β :: Float64
end

function rand(rng::AbstractRNG, dist::BetaGeometric)
  p = rand(rng, Beta(dist.α, dist.β))
  rand(rng, Geometric(p))
end

# https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/bgepdf.htm
function logpdf(dist::BetaGeometric, n)
  logbeta(dist.α + 1, dist.β + n) - logbeta(dist.α, dist.β)
end

function logvarpdf(dist::BetaGeometric, n)
  p = exp(digamma(dist.α) - logaddexp(digamma(dist.α), digamma(dist.β)))
  logpdf(Geometric(p), 2)
end

function add_obs!(dist::BetaGeometric, n, pscount)
  dist.α += pscount
  dist.β += n*pscount
  dist
end



function calkin_wilf_children(x)
  a = numerator(x)
  b = denominator(x)
  return [a // (a+b), (a+b) // b]
end

using Memoize: @memoize
@memoize function ratios_of_calkin_wilf_level(i)
  if i == 0
    [1//1]
  else
    mapreduce(calkin_wilf_children,append!,ratios_of_calkin_wilf_level(i-1))
  end
end

@memoize function proper_ratios_of_calkin_wilf_level(i)
  Set(filter(x->x<1, ratios_of_calkin_wilf_level(i)))
end

calkin_wilf_level(x::Rational) = stern_brocot_level(x)

# algorithm from https://en.wikipedia.org/wiki/Stern%E2%80%93Brocot_tree
function stern_brocot_path(x::Rational)
  @assert 0 < x
  path = Bool[]
  l = 0//1 # lower bound
  h = 1//0 # higher bound
  while true
    m = (numerator(l) + numerator(h)) // (denominator(l) + denominator(h))
    if x < m
      push!(path, false)
      h = m
    elseif x > m
      push!(path, true)
      l = m
    else
      break
    end
  end
  return path
end

function stern_brocot_level(x::Rational)
  @assert 0 < x
  level = 0
  l = 0//1 # lower bound
  h = 1//0 # higher bound
  while true
    m = (numerator(l) + numerator(h)) // (denominator(l) + denominator(h))
    if x < m
      level += 1
      h = m
    elseif x > m
      level += 1
      l = m
    else
      break
    end
  end
  return level
end





@probprog function calkin_wilf_product_model(params, nt)
  harmony_nt, rhythm_nt = nt
  harmony_rule ~ params.harmony_cond[harmony_nt]
  if "start" ⊣ rhythm_nt
    rhythm_rule ~ Dirac(rhythm_start_rule)
  elseif arity(harmony_rule) == 1
    rhythm_rule ~ Dirac(rhythm_termination)
  else
    levelm1 ~ params.level_dist
    level = levelm1 + 1
    level_ratios = proper_ratios_of_calkin_wilf_level(level)
    ratio ~ UniformCategorical(level_ratios)
    rhythm_rule ~ Dirac(rhythm_split_rule(ratio))
  end
  return
end

import SimpleProbabilisticPrograms: fromtrace, totrace
function fromtrace(::typeof(calkin_wilf_product_model), trace)
  ProductRule(trace.harmony_rule, trace.rhythm_rule)
end
function totrace(::typeof(calkin_wilf_product_model), rule)
  harmony_rule = rule[1]
  rhythm_rule  = rule[2]
  if rhythm_rule in (rhythm_start_rule, rhythm_termination)
    (; harmony_rule, rhythm_rule) # named tuple shorthand notation
  else
    ratio = rule[2].ratio
    levelm1 = calkin_wilf_level(ratio) - 1
    (; harmony_rule, levelm1, ratio, rhythm_rule)
  end
end

function calkin_wilf_product_prior()
  harmony_dist(nt) = flat_dircat(applicable_rules(rules, nt))
  (harmony_cond = Dict(nt => harmony_dist(nt) for nt in [nts; START]),
   level_dist  = BetaGeometric(1, 10_000_000))
end

params = calkin_wilf_product_prior()
nt = (nts[31], nonterminal_category(1//2))
model = calkin_wilf_product_model(params, nt)
rule = rand(model)
logpdf(model, rule)



@probprog function product_rule_model(params, nt)
  harmony_nt, rhythm_nt = nt
  harmony_rule ~ params.harmony_cond[harmony_nt]
  if "start" ⊣ rhythm_nt
    rhythm_rule ~ Dirac(rhythm_start_rule)
  elseif arity(harmony_rule) == 1
    rhythm_rule ~ Dirac(rhythm_termination)
  else
    rhythm_rule ~ params.rhythm_dist
  end
  return
end

import SimpleProbabilisticPrograms: fromtrace, totrace
fromtrace(::typeof(product_rule_model), trace) = ProductRule(trace...)
totrace(::typeof(product_rule_model), rule) = 
  (harmony_rule=rule[1], rhythm_rule=rule[2])

function product_rule_prior()
  harmony_dist(nt) = flat_dircat(applicable_rules(rules, nt))
  (harmony_cond = Dict(nt => harmony_dist(nt) for nt in [nts; START]),
   rhythm_dist  = flat_dircat(collect(split_rules)))
end

logpdf(grammar::ProductGrammar, lhs, rule) = logpdf(rule_dist(grammar, lhs), rule)
logvarpdf(grammar::ProductGrammar, lhs, rule) = logvarpdf(rule_dist(grammar, lhs), rule)

function calkin_wilf_sequence(max_level)
  n = sum(2^l for l in 0:max_level)
  seq = zeros(Rational{Int}, n)
  seq[1] = 1
  for i in 2:n
    seq[i] = 1 // (2*floor(seq[i-1]) - seq[i-1] + 1)
  end
  return seq
end

using Random: randperm, default_rng
# k-fold cross validation for n data points
function cross_validation_index_split(num_folds, num_total, rng=default_rng())
  num_perfold = ceil(Int, num_total/num_folds)
  num_lastfold = num_total - (num_folds-1) * num_perfold
  fold_lenghts = [fill(num_perfold, num_folds-1); num_lastfold]
  fold_ends = accumulate(+, fold_lenghts)
  fold_starts = fold_ends - fold_lenghts .+ 1
  shuffled_idxs = randperm(rng, num_total)
  test_indices = [shuffled_idxs[i:j] for (i,j) in zip(fold_starts,fold_ends)]
  train_indices = [setdiff(1:num_total, idxs) for idxs in test_indices]
  return collect(zip(test_indices, train_indices))
end

harmony_grammar = StdGrammar([START], rules, nothing)

split_ratios = filter(x->x<1, calkin_wilf_sequence(12))
split_rules = Set(rhythm_split_rule.(split_ratios))
rhythm_rules = union(split_rules, [rhythm_start_rule, rhythm_termination])
rhythm_grammar = RhythmGrammar(rhythm_rules, nothing)

prod_seqs = product_sequence.(treebank)
start = (START, rhythm_start_category)


epochs = 2
n = 150
k = 10
accs = zeros(epochs, n)
test_train_idx_pairs = cross_validation_index_split(k, n)
@showprogress for (test_idxs, train_idxs) in test_train_idx_pairs
  g = ProductGrammar(harmony_grammar, rhythm_grammar, product_rule_prior())
  for e in 1:epochs
    variational_inference_step!(
      g, prod_seqs[train_idxs], start, product_rule_prior)
    accs[e, test_idxs] = calc_accs(
      g, treebank[test_idxs], start,treekey="product_tree")
    println("mean acc: ", sum(accs[e, test_idxs]) / length(accs[e, test_idxs]))
  end
end
accs[end,:] |> mean

treebank_split_ratios = mapreduce(append!, treebank) do tune
  @_ tree2derivation(treelet2rhythmrule, tune["rhythm_tree"]) |>
     filter("split" ⊣ _, __) |>
     map(_.ratio, __)
end
treebank_split_ratio_counts = 
  @_ counter(treebank_split_ratios) |>
     collect |>
     sort(__, by=_[2], rev=true)
foreach(println, first(treebank_split_ratio_counts, 20))

rule_dist(g::ProductGrammar, lhs) = product_rule_model(g.params, lhs)
g = ProductGrammar(harmony_grammar, rhythm_grammar, product_rule_prior())
@time for e in 1:2
  variational_inference_step!(
    g, prod_seqs[1:150], start, product_rule_prior)
  println(mean(calc_accs(
    g, treebank[1:150], start,treekey="product_tree")))
end

@time apps = mapreduce(append!, prod_seqs) do seq
  chart = chartparse(g, BestDerivationScoring(), seq)
  chart[1, length(seq)][start].apps
end
predicted_split_ratio_counts = 
  @_ apps |> 
     map(_.rule[2], __) |> 
     filter("split" ⊣ _, __) |>
     map(_.ratio, __) |>
     counter |>
     collect |>
     sort(__, by=_[2], rev=true)
foreach(println, first(predicted_split_ratio_counts, 20))

rule_dist(g::ProductGrammar, lhs) = calkin_wilf_product_model(g.params, lhs)
g = ProductGrammar(harmony_grammar, rhythm_grammar, calkin_wilf_product_prior())
@time for e in 1:4
  variational_inference_step!(
    g, prod_seqs[1:150], start, calkin_wilf_product_prior)
  println(mean(calc_accs(
    g, treebank[1:150], start,treekey="product_tree")))
end

g.params.level_dist

@time apps = mapreduce(append!, prod_seqs) do seq
  chart = chartparse(g, BestDerivationScoring(), seq)
  chart[1, length(seq)][start].apps
end
predicted_split_ratio_counts = 
  @_ apps |> 
     map(_.rule[2], __) |> 
     filter("split" ⊣ _, __) |>
     map(_.ratio, __) |>
     counter |>
     collect |>
     sort(__, by=_[2], rev=true)
foreach(println, first(predicted_split_ratio_counts, 20))



@time accs = calc_accs(grammar, treebank, START)
sum(accs) / length(accs)

@time accs = let start = rhythm_start_category 
  calc_accs(rhythm_grammar, treebank, start, treekey="rhythm_tree")
end
sum(accs) / length(accs)

@time accs = let start = (START, rhythm_start_category)
  calc_accs(product_grammar, treebank, start, treekey="product_tree")
end
sum(accs) / length(accs)





