###############
### Imports ###
###############

begin
  using AbstractGrammars
  using SimpleProbabilisticPrograms
  using Pitches: Pitches

  # imports for overloading
  # import Base: *, +
  # import AbstractGrammars: default
  # import Distributions: logpdf
  import Distributions: Geometric

  # imports without overloading
  # using DataStructures: counter, Accumulator
  using DataStructures: counter
  using Statistics: mean
  using Underscores: @_
  using ProgressMeter: Progress, progress_map, @showprogress
end

#############
### Utils ###
#############

begin
  *(a::Accumulator, n::Number) = Accumulator(Dict(k => v*n for (k,v) in a.map))
  *(n::Number, a::Accumulator) = Accumulator(Dict(k => n*v for (k,v) in a.map))
  +(a::Accumulator, n::Number) = Accumulator(Dict(k => v+n for (k,v) in a.map))
  +(n::Number, a::Accumulator) = Accumulator(Dict(k => n+v for (k,v) in a.map))

  function most_frequent(a::Accumulator, n=10)
    @_ collect(a.map) |>
      sort!(__, by=kv->kv[2], rev=true) |>
      first(__, n)
  end
end

function predict_derivations(
    grammar, ruledist, sequences, seq2start; showprogress=true
  )
  scoring = BestDerivationScoring(ruledist)
  p = Progress(
    length(treebank);
    desc="calculating derivations: ", 
    enabled=showprogress
  )
  progress_map(sequences; progress=p) do seq
    chart = chartparse(grammar, scoring, seq)
    start = seq2start(seq)
    apps = chart[1, end][start].apps
    (start, [app.rule for app in apps])
  end
end

function calc_accs(
    grammar, ruledist, treebank, seq2start; 
    treekey="harmony_tree", showprogress=true
  )
  trees = getindex.(treebank, treekey)
  sequences = leaflabels.(trees)
  derivations = predict_derivations(
    grammar, ruledist, sequences, seq2start; showprogress
  )
  map(trees, derivations) do tree, (start, derivation)
    tree_similarity(tree, apply(derivation, start))
  end
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

#####################
### Load Treebank ###
#####################

begin
  include("jazz_learnability/JazzTreebank.jl")
  # TPCC ... tonal pitch-class chord
  using .JazzTreebank: Chord, ChordForm, TPCC, parse_chord, load_tunes_and_treebank
  tunes, treebank = load_tunes_and_treebank();
end

#######################
### Harmony grammar ###
#######################

const all_chords = collect(
  Chord(Pitches.parsespelledpitch(letter * acc), form) 
  for letter in 'A':'G'
  for acc in ("b", "#", "")
  for form in instances(ChordForm))

function mk_harmony_grammar(rulekinds=[:duplication, :leftheaded, :rightheaded])
  ts  = T.(all_chords)  # terminals
  nts = NT.(all_chords) # nonterminals
  
  # termination rules are always included
  rules = [nt --> t for (nt, t) in zip(nts, ts)]

  # include other rule kinds
  if :duplication in rulekinds
    append!(rules, [nt --> (nt, nt) for nt in nts])
  end
  distinct_pairs(xs) = [(x1, x2) for x1 in xs for x2 in xs if x1 != x2]
  if :leftheaded in rulekinds
    append!(rules, [rhs[1] --> rhs for rhs in distinct_pairs(nts)])
  end
  if :rightheaded in rulekinds
    append!(rules, [rhs[2] --> rhs for rhs in distinct_pairs(nts)])
  end

  start_categories = nts
  StdGrammar(Set(start_categories), Set(rules))
end

# test harmony grammar
@time begin
  grammar = mk_harmony_grammar([:duplication, :rightheaded])
  ruledist = symdircat_ruledist(NT.(all_chords), grammar.rules, 0.1)
  accs = calc_accs(
    grammar, ruledist, treebank, seq->NT(seq[end]), treekey="harmony_tree")
  @show mean(accs)
  harmony_trees = [tune["harmony_tree"] for tune in treebank]
  observe_trees!(treelet2stdrule, ruledist, harmony_trees)
  accs = calc_accs(
    grammar, ruledist, treebank, seq->NT(seq[end]), treekey="harmony_tree")
  @show mean(accs)
end

############################
### Test with dummy data ###
############################

# terminalss = collect([H.terminal_cat(c)]
#   for c in [Chord(p"C", MAJ7), Chord(p"G", DOM), Chord(p"C", MAJ7)])
# terminalss = fill([terminal_category(Chord(p"C", MAJ7))], 50)

# terminals = [T(rand(all_chords)) for _ in 1:50]
# scoring = WDS(grammar) # weighted derivation scoring
# @time chart = chartparse(grammar, scoring, terminals)
# forest = chart[1, length(terminals)][NT(terminals[end])]
# @time sample_derivations(scoring, forest, 1) .|> (app -> arity(app.rule))

######################
### Rhythm Grammar ###
######################

begin
  import AbstractGrammars: arity, apply, push_completions!

  const RhythmCategory = StdCategory{Rational{Int}}

  # unary termination rules and binary split rules
  struct RhythmRule <: Rule{RhythmCategory}
    istermination :: Bool
    split_ratio   :: Rational{Int}
  end

  RhymTerm() = RhythmRule(true, default(Rational{Int}))
  RhymSplit(split_ratio) = RhythmRule(false, split_ratio)

  function show(io::IO, r::RhythmRule)
    if r.istermination
      print(io, "RhymTerm()")
    else
      print(io, "RhymSplit($(r.split_ratio))")
    end
  end

  arity(r::RhythmRule) = r.istermination ? 1 : 2

  function apply(r::RhythmRule, c::RhythmCategory)
    if isterminal(c)
      nothing
    elseif r.istermination
      tuple(T(c))
    else # r is a split rule
      tuple(NT(r.split_ratio * c.val), NT((1 - r.split_ratio) * c.val))
    end
  end

  mutable struct RhythmGrammar <: Grammar{RhythmRule}
    splitrules  :: Set{RhythmRule}
    function RhythmGrammar(splitrules)
      @assert all(arity(r) == 2 for r in splitrules)
      new(Set(splitrules))
    end
  end

  function push_completions!(::RhythmGrammar, stack, c)
    if isterminal(c)
      push!(stack, App(NT(c), RhymTerm()))
    end
  end

  function push_completions!(grammar::RhythmGrammar, stack, c1, c2)
    if isnonterminal(c1) && isnonterminal(c2)
      s = sum(c1.val + c2.val)
      ratio = c1.val / s
      rule = RhymSplit(ratio)
      if rule in grammar.splitrules
        push!(stack, App(NT(s), rule))
      end
    end
  end

  function mk_rhythm_grammar(max_num=100, max_denom=100)
    splitrules = Set(RhymSplit(n//d) for n in 1:max_num for d in n+1:max_denom)
    RhythmGrammar(splitrules)
  end

  function treelet2rhythmrule(treelet)
    root = treelet.root_label
    children = treelet.child_labels
    if arity(treelet) == 1
      child = children[1]
      if isnonterminal(root) && isterminal(child) && root.val == child.val
        RhymTerm()
      else
        error("cannot convert unary $treelet into a rhythm rule")
      end
    elseif arity(treelet) == 2 && 
          isnonterminal(root) && 
          isnonterminal(children[1]) && isnonterminal(children[2]) &&
          root.val == children[1].val + children[2].val
      RhymSplit(children[1].val // root.val)
    else
      error("cannot convert binary $treelet into a rhythm rule")
    end
  end
end

begin # test rhythm grammar single sequence
  tune = treebank[30]; tune["title"]
  terminals = T.(normalize(Rational.(JazzTreebank.chord_durations(tune))))
  rhythm_grammar = mk_rhythm_grammar()
  rhythm_dircat = symdircat([rhythm_grammar.splitrules; RhymTerm()], 0.1)
  ruledist = _ -> rhythm_dircat
  scoring = WDS(ruledist, rhythm_grammar)
  @time chart = chartparse(rhythm_grammar, scoring, terminals)
  chart[1, end][NT(1//1)]
end

# test rhythm grammar
@time begin
  rhythm_grammar = mk_rhythm_grammar()
  rhythm_dircat = symdircat(union(rhythm_grammar.splitrules, [RhymTerm()]), 0.1)
  ruledist = _ -> rhythm_dircat
  accs = calc_accs(rhythm_grammar, ruledist, treebank, seq -> NT(1//1), treekey="rhythm_tree")
  @show mean(accs)
  for tune in treebank
    observe_tree!(treelet2rhythmrule, ruledist, tune["rhythm_tree"])
  end
  accs = calc_accs(rhythm_grammar, ruledist, treebank, seq -> NT(1//1), treekey="rhythm_tree")
  @show mean(accs)
end

#######################
### Product grammar ###
#######################

begin
  function treelet2prodrule(treelet2fstrule, treelet2sndrule)
    function unzip(xs)
      n = length(first(xs))
      ntuple(i -> map(x -> x[i], xs), n)
    end
    function (treelet)
      lhs1, lhs2 = treelet.root_label
      rhs1, rhs2 = unzip(treelet.child_labels)
      rule1 = treelet2fstrule(Treelet(lhs1, rhs1...))
      rule2 = treelet2sndrule(Treelet(lhs2, rhs2...))
      ProductRule(rule1, rule2)
    end
  end

  @probprog function simple_product_model(harmony_dist, rhythm_dist, nt)
    harmony_nt, _rhythm_nt = nt
    harmony_rule ~ harmony_dist(harmony_nt)
    if arity(harmony_rule) == 1
      rhythm_rule ~ Dirac(RhymTerm())
    else
      rhythm_rule ~ rhythm_dist
    end
    return
  end

  import SimpleProbabilisticPrograms: fromtrace, totrace
  fromtrace(::typeof(simple_product_model), trace) = ProductRule(trace...)
  totrace(::typeof(simple_product_model), rule) = 
    (harmony_rule=rule[1], rhythm_rule=rule[2])

  function mk_simple_product_prior(harmony_grammar, rhythm_grammar)
    nonterminals = NT.(all_chords)
    harmony_dist = symdircat_ruledist(nonterminals, harmony_grammar.rules, 0.1)
    rhythm_dist = symdircat(rhythm_grammar.splitrules, 0.1)
    nt -> simple_product_model(harmony_dist, rhythm_dist, nt)
  end
end

# test simple product grammar
@time begin
  rulekinds = [:duplication, :rightheaded]
  hg = mk_harmony_grammar(rulekinds)
  rg = mk_rhythm_grammar()
  pg = ProductGrammar(hg, rg)
  ruledist = mk_simple_product_prior(hg, rg)
  seq2start = seq -> (NT(seq[end][1]), NT(1//1))
  accs = calc_accs(pg, ruledist, treebank, seq2start, treekey="product_tree")
  @show mean(accs)
  prod_trees = [tune["product_tree"] for tune in treebank]
  treelet2rule = treelet2prodrule(treelet2stdrule, treelet2rhythmrule)
  observe_trees!(treelet2rule, ruledist, prod_trees)
  accs = calc_accs(pg, ruledist, treebank, seq2start, treekey="product_tree")
  @show mean(accs)
end

# test variational inference
begin 
  rulekinds = [:duplication, :rightheaded]
  hg = mk_harmony_grammar(rulekinds)
  rg = mk_rhythm_grammar()
  pg = ProductGrammar(hg, rg)
  ruledist = mk_simple_product_prior(hg, rg)
  seq2start = seq -> (NT(seq[end][1]), NT(1//1))
  sequences = [leaflabels(tune["product_tree"]) for tune in treebank]
  counts = estimate_rule_counts(ruledist, pg, sequences[30:32], seq2start)
  @_ collect(counts) |> 
    filter(arity(_[1].rule) == 2, __) |> 
    sort(__, by=_[2], rev=true) |> 
    map((_[1].rule, _[2]), __) |>
    first(__, 10) |>
    foreach(println, __)

  mk_prior = () -> mk_simple_product_prior(hg, rg)
  ruledist_post = runvi(2, mk_prior, pg, sequences, seq2start)
  accs = calc_accs(pg, ruledist_post, treebank, seq2start, treekey="product_tree")
  mean(accs)
end

#########################
### Calkin-Wilf model ###
#########################

max_lvl = 8
k = 300
Dict(l => k * (max_lvl-l+1) for l in 1:max_lvl)
DirCat(Dict(r => k * (max_lvl-calkin_wilf_level(r)+1) for r in calkin_wilf_sequence(max_lvl)))

include("jazz_learnability/calkin_wilf.jl")

function mk_calkin_wilf_product_prior(harmony_grammar, lvl_accept, max_lvl)
  harmony_dist = symdircat_ruledist(NT.(all_chords), harmony_grammar.rules, 0.1)
  # level_dist = BetaGeometric(1, k)
  # level_dist = symdircat(1:max_lvl, k)
  # level_dist = DirCat(Dict(l => k * (max_lvl-l+1) for l in 1:max_lvl))
  level_dist = Geometric(1-lvl_accept)
  ratio_dists = symdircat.(proper_ratios_of_calkin_wilf_level.(1:max_lvl), 0.1)
  # nt -> calkin_wilf_product_model(harmony_dist, level_dist, ratio_dists, nt)
  # ratio_dist = DirCat(Dict(r => k * (max_lvl-calkin_wilf_level(r)+1) for r in calkin_wilf_sequence(max_lvl)))
  nt -> calkin_wilf_product_model(harmony_dist, level_dist, ratio_dists, nt)
end

@probprog function calkin_wilf_product_model(
    harmony_dist, level_dist, ratio_dists, nt
  )
  harmony_nt, _rhythm_nt = nt
  harmony_rule ~ harmony_dist(harmony_nt)
  if arity(harmony_rule) == 1
    rhythm_rule ~ Dirac(RhymTerm())
  else
    levelm1 ~ level_dist # level minus one
    level = levelm1 + 1
    # level_ratios = proper_ratios_of_calkin_wilf_level(level)
    # ratio ~ UniformCategorical(level_ratios)

    # level ~ level_dist
    ratio ~ ratio_dists[level]
    
    rhythm_rule ~ Dirac(RhymSplit(ratio))
  end
  return
end

import SimpleProbabilisticPrograms: fromtrace, totrace
function fromtrace(::typeof(calkin_wilf_product_model), trace)
  ProductRule(trace.harmony_rule, trace.rhythm_rule)
end
function totrace(::typeof(calkin_wilf_product_model), rule)
  harmony_rule, rhythm_rule = rule[1], rule[2]
  if rhythm_rule.istermination
    (; harmony_rule, rhythm_rule) # named tuple shorthand notation
  else
    ratio = rhythm_rule.split_ratio
    levelm1 = calkin_wilf_level(ratio) - 1
    (; harmony_rule, levelm1, ratio, rhythm_rule)
    # level = calkin_wilf_level(ratio)
    # (; harmony_rule, level, ratio, rhythm_rule)
    # (; harmony_rule, ratio, rhythm_rule)
  end
end

# test calkin wilf model
begin 
  rulekinds = [:duplication, :rightheaded]
  hg = mk_harmony_grammar(rulekinds)
  max_level = 8
  split_ratios = filter(x -> x < 1, calkin_wilf_sequence(max_level))
  rg = RhythmGrammar(Set(RhymSplit.(split_ratios)))
  pg = ProductGrammar(hg, rg)
  ruledist = mk_calkin_wilf_product_prior(hg, 0.5, max_level)
  seq2start = seq -> (NT(seq[end][1]), NT(1//1))
  sequences = [leaflabels(tune["product_tree"]) for tune in treebank]
end

counts = estimate_rule_counts(ruledist, pg, sequences, seq2start)
@_ collect(counts) |> 
  filter(arity(_[1].rule) == 2, __) |> 
  sort(__, by=_[2], rev=true) |> 
  map((_[1].rule, _[2]), __) |>
  first(__, 30) |>
  foreach(println, __)

using DataStructures: SortedDict
for level_accept in (0.75:0.01:0.80)
  mk_prior = () -> mk_calkin_wilf_product_prior(hg, level_accept, max_level)
  ruledist_post = runvi(5, mk_prior, pg, sequences, seq2start, showprogress=false)
  accs = calc_accs(pg, ruledist_post, treebank, seq2start, treekey="product_tree", showprogress=false)
  println("level accept $level_accept | mean(accs) = $(mean(accs))")
end

println(SortedDict(ruledist_post.ratio_dists[7].pscounts))

##################################
### Compare split ratio counts ###
##################################

best_derivations = begin
  sequences = [leaflabels(tune["product_tree"]) for tune in treebank]
  derivations = predict_derivations(
    pg, ruledist_post, sequences, seq2start, showprogress=false
  )
end
predicted_split_ratios = [
  rule[2].split_ratio 
  for (start, derivation) in best_derivations 
  for rule in derivation 
  if !rule[2].istermination
]
predicted_split_ratio_counts = 
  @_ counter(predicted_split_ratios) |>
     collect |>
     sort(__, by=_[2], rev=true)
foreach(println, first(predicted_split_ratio_counts, 20))

treebank_split_ratios = mapreduce(append!, treebank) do tune
  @_ tree2derivation(treelet2rhythmrule, tune["rhythm_tree"]) |>
     filter(!_.istermination, __) |>
     map(_.split_ratio, __)
end
treebank_split_ratio_counts = 
  @_ counter(treebank_split_ratios) |>
     collect |>
     sort(__, by=_[2], rev=true)
foreach(println, first(treebank_split_ratio_counts, 20))