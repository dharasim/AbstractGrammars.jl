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

const TPCC = Chord{Pitch{SpelledIC}} # Tonal Pitch-Class Chord

function title_and_tree(tune)
  remove_asterisk(label::String) = replace(label, "*" => "")

  function transform(tree)
    function categorize(tree)
      if isleaf(tree)
        Tree(nonterminal_category(tree.label), Tree(terminal_category(tree.label)))
      else
        Tree(nonterminal_category(tree.label), map(categorize, tree.children))
      end
    end
    Tree(start_category(TPCC), categorize(tree))
  end

  tree = @_ tune["trees"][1]["open_constituent_tree"] |> 
    dict2tree(remove_asterisk, __) |>
    map(parse_chord, __) |>
    transform(__)

  (title = tune["title"], tree = tree)
end

treebank_url = "https://raw.githubusercontent.com/DCMLab/JazzHarmonyTreebank/master/treebank.json"
tunes = HTTP.get(treebank_url).body |> String |> JSON.parse
treebank = @_ tunes |> filter(haskey(_, "trees"), __) |> map(title_and_tree, __)

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
foreach(tune -> observe_tree!(grammar.params, tune.tree), treebank)

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
    tree = treebank[i].tree
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

################################################################################

