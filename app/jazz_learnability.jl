using AbstractGrammars

# imports for overloading
import AbstractGrammars: default
import Distributions: logpdf

# imports without overloading
using AbstractGrammars.ConjugateModels: DirCat, add_obs!
using Pitches: parsespelledpitch, Pitch, SpelledIC, MidiIC, midipc, alteration, @p_str, tomidi
using Underscores: @_

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

function preprocess_tree!(tune)
  remove_asterisk(label::String) = replace(label, "*" => "")

  function categorize_and_insert_unary_rules(tree)
    function categorize(tree)
      if isleaf(tree)
        Tree(nonterminal_category(tree.label), Tree(terminal_category(tree.label)))
      else
        Tree(nonterminal_category(tree.label), map(categorize, tree.children))
      end
    end
    Tree(start_category(TPCC), categorize(tree))
  end

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
flat_dircat(xs) = DirCat(Dict(x => 1 for x in xs))
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

# using Distributed
# using SharedArrays

# addprocs(6)
# workers()
# @everywhere using AbstractGrammars

function calc_accs(grammar, treebank, startsymbol)
  scoring = BestDerivationScoring()
  accs = zeros(length(treebank))
  for i in eachindex(treebank)
    print(i, ' ', treebank[i].title, ' ')
    tree = treebank[i]["tree"]
    terminalss = [[c] for c in leaflabels(tree)]
    chart = chartparse(grammar, scoring, terminalss)
    apps = chart[1, length(terminalss)][startsymbol].apps
    derivation = [app.rule for app in apps]
    accs[i] = tree_similarity(tree, apply(derivation, startsymbol))
    println(accs[i])
  end
  return accs
end

@time accs = calc_accs(grammar, treebank[1:150], START)
sum(accs) / length(accs)

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

  return relabel(tune["tree"])
end

################################################################################

@time chord_durations(tune)
@time leaf_durations(tune)
@time normalized_duration_tree.(treebank)
@time chord_durations.(tunes);

failed = 0
for tune in tunes
  try
    chord_durations(tune)
  catch
    failed += 1
    println(tune["title"])
  end
end
failed