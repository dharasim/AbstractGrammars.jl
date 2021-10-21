using AbstractGrammars
using AbstractGrammars.GeneralCategories

# imports for overloading
import AbstractGrammars: default
import Distributions: logpdf

# imports without overloading
using AbstractGrammars.ConjugateModels: DirCat, add_obs!
using Pitches: parsespelledpitch, Pitch, SpelledIC, MidiIC, midipc, alteration, @p_str, tomidi
using Underscores: @_

# named imports
import AbstractGrammars.Headed
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

##################################
### General context-free rules ###
##################################

using AbstractGrammars.AtMosts

struct CFRule{C} <: AbstractRule{C}
  lhs :: C
  rhs :: AtMost2{C}
end

CFRule(lhs, rhs...) = CFRule(lhs, AtMost2(rhs...))
arity(r::CFRule) = length(r.rhs)
# @assert [CFRule('a', 'b', 'c'), CFRule('d', 'e')] |> eltype |> isbitstype

function derivation(tree::Tree{T}) where T
  rules = [CFRule(start_category(T), nonterminal_category(tree.label))]

  push_rules(tree::Leaf) = begin 
    r = CFRule(nonterminal_category(tree.label), terminal_category(tree.label))
    push!(rules, r)
  end
  push_rules(tree::Binary) = begin
    r = CFRule(
      nonterminal_category(tree.label), 
      nonterminal_category(tree.left.label), 
      nonterminal_category(tree.right.label))
    push!(rules, r)
    push_rules(tree.left)
    push_rules(tree.right)
  end

  push_rules(tree)
  return rules
end

function derivation2tree(grammar, derivation::Vector{App{C, R}}) where {C, R}
  i = 0 # rule index
  next_app() = (i += 1; derivation[i])
  
  function rewrite(nt)
    app = next_app()
    @assert nt == app.lhs
    rhs = apply(grammar, app)
    if length(rhs) == 1 # terminal rule
      Leaf(rhs[1])
    elseif length(rhs) == 2 # binary rule
      Binary(nt, rewrite(rhs[1]), rewrite(rhs[2]))
    else
      error("only binary rules and unary termination rules are supported")
    end
  end

  app1, app2 = derivation[1:2]
  if length(apply(grammar, app1)) == 1
    # if initial rule is unary, then skip it for the tree
    i += 1
    rewrite(app2.lhs)
  else
    rewrite(app1.lhs)
  end
end

function cfrule_to_headedrule_app(r::CFRule{Headed.Category{T}}) where T
    if arity(r) == 1 && "start" ⊣ r.lhs && "nonterminal" ⊣ r.rhs[1]
      return App(r.lhs, Headed.start_rule(r.rhs[1]))
    elseif arity(r) == 1 && "nonterminal" ⊣ r.lhs && "terminal" ⊣ r.rhs[1]
      return App(r.lhs, Headed.termination_rule(T))
    elseif arity(r) == 2 && "nonterminal" ⊣ (r.lhs, r.rhs...)
      if r.lhs == r.rhs[1] == r.rhs[2]
        return App(r.lhs, Headed.duplication_rule(T))
      elseif r.lhs == r.rhs[1] && r.rhs[1] != r.rhs[2]
        return App(r.lhs, Headed.leftheaded_rule(r.rhs[2]))
      elseif r.lhs == r.rhs[2] && r.rhs[1] != r.rhs[2]
        return App(r.lhs, Headed.rightheaded_rule(r.rhs[1]))
      end
    end
    error("$r could not be converted into a headed rule")
end

#####################
### Read treebank ###
#####################

function title_and_tree(tune)
  remove_asterisk(label::String) = replace(label, "*" => "")
  (title = tune["title"], 
   tree = @_ tune["trees"][1]["open_constituent_tree"] |> 
             dict2tree(remove_asterisk, __) |>
             map(parse_chord, __))
end

treebank_url = "https://raw.githubusercontent.com/DCMLab/JazzHarmonyTreebank/master/treebank.json"
tunes = HTTP.get(treebank_url).body |> String |> JSON.parse
treebank = @_ tunes |> filter(haskey(_, "trees"), __) |> map(title_and_tree, __)

#########################
### Construct grammar ###
#########################

const TPCC = Chord{Pitch{SpelledIC}} # Tonal Pitch-Class Chord

all_chords = collect(
  Chord(parsespelledpitch(letter * acc), form) 
  for letter in 'A':'G'
  for acc in ("b", "#", "")
  for form in instances(ChordForm))

startsym = start_category(TPCC)
ts       = terminal_category.(all_chords)
nts      = nonterminal_category.(all_chords)

start_rules = Set(Headed.start_rule.(nts))
nonstart_rules = Set([
  Headed.termination_rule(TPCC);
  Headed.duplication_rule(TPCC);
  Headed.leftheaded_rule.(nts);
  Headed.rightheaded_rule.(nts)])

rules = union(start_rules, nonstart_rules)

# probability model
flat_dircat(xs) = DirCat(Dict(x => 1 for x in xs))
params = (
  start_dist = flat_dircat(start_rules),
  nonstart_dists = Dict(nt => flat_dircat(nonstart_rules) for nt in nts))

function logpdf(g::Headed.Grammar, lhs, rule)
  if "start" ⊣ lhs && "start" ⊣ rule
    logpdf(g.params.start_dist, rule)
  elseif "nonterminal" ⊣ lhs && !("startrule" ⊣ rule)
    logpdf(g.params.nonstart_dists[lhs], rule)
  else
    log(0)
  end
end

# supervised training by observation of trees
function observe_tree!(params, tree)
  apps = cfrule_to_headedrule_app.(derivation(tree))
  for app in apps
    if "start" ⊣ app.rule
      add_obs!(params.start_dist, app.rule, 1)
    else
      add_obs!(params.nonstart_dists[app.lhs], app.rule, 1)
    end
  end
end

foreach(tune -> observe_tree!(params, tune.tree), treebank)
grammar = Headed.Grammar(rules, params)

############################
### Test with dummy data ###
############################

# terminalss = collect([H.terminal_cat(c)]
#   for c in [Chord(p"C", MAJ7), Chord(p"G", DOM), Chord(p"C", MAJ7)])
terminalss = fill([terminal_category(Chord(p"C", MAJ7))], 50)
scoring = WDS(grammar) # weighted derivation scoring
@time chart = chartparse(grammar, scoring, terminalss)
@time sample_derivations(scoring, chart[1,length(terminalss)][startsym], 1) .|> 
  (app -> app.rule.tag)

##########################
### Test with treebank ###
##########################

scoring = BestDerivationScoring()
treebank = treebank[1:10]
accs = zeros(length(treebank))
@time for i in eachindex(treebank)
  print(i, ' ', treebank[i].title, ' ')
  tree = treebank[i].tree
  terminalss = [[terminal_category(c)] for c in leaflabels(tree)]
  chart = chartparse(grammar, scoring, terminalss)
  apps = chart[1, length(terminalss)][startsym].apps
  accs[i] = tree_similarity(tree, derivation2tree(grammar, apps))
  println(accs[i])
end
sum(accs) / length(accs)

################################################################################

