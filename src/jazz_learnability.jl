import HTTP, JSON

import AbstractGrammars as AG
import AbstractGrammars.HeadedSimple as H

import AbstractGrammars: default, ⊣
import Base: map

using Test

using Pitches: parsespelledpitch, Pitch, SpelledIC, MidiIC, midipc, alteration, @p_str, tomidi
using Underscores: @_

#############
### Utils ###
#############

default(::Type{SpelledIC}) = SpelledIC(0)
default(::Type{Pitch{I}}) where I = Pitch(default(I))

#######################
### Trees and Tunes ###
#######################

abstract type Tree{L} end
struct Binary{L} <: Tree{L}
  label :: L
  left  :: Tree{L}
  right :: Tree{L}
end
struct Leaf{L} <: Tree{L}
  label :: L
end

map(f, tree::Leaf) = Leaf(f(tree.label))
map(f, tree::Binary) = 
  Binary(f(tree.label), map(f, tree.left), map(f, tree.right))

function dict2tree(f, dict)
  if isempty(dict["children"])
    Leaf{String}(f(dict["label"]))
  else
    @assert length(dict["children"]) == 2
    Binary{String}(
      f(dict["label"]), 
      dict2tree(f, dict["children"][1]), 
      dict2tree(f, dict["children"][2]) )
  end
end

function title_and_tree(tune)
  remove_asterisk(label::String) = replace(label, "*" => "")
  (title = tune["title"], 
   tree = @_ tune["trees"][1]["open_constituent_tree"] |> dict2tree(remove_asterisk, __))
end

function innerlabels(tree::Tree{L}) where L
  labels = L[]

  pushlabels(tree::Binary) = begin
    push!(labels, tree.label)
    pushlabels(tree.left)
    pushlabels(tree.right)
  end
  pushlabels(::Leaf) = nothing

  pushlabels(tree)
  labels
end

function leaflabels(tree::Tree{L}) where L
  labels = L[]

  pushlabels(tree::Binary) = (pushlabels(tree.left); pushlabels(tree.right))
  pushlabels(tree::Leaf) = push!(labels, tree.label);

  pushlabels(tree)
  labels
end

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

@test all(instances(ChordForm)) do form
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

############
### Main ###
############

treebank_url = "https://raw.githubusercontent.com/DCMLab/JazzHarmonyTreebank/master/treebank.json"
tunes = HTTP.get(treebank_url).body |> String |> JSON.parse
treebank = @_ tunes |> filter(haskey(_, "trees"), __) |> map(title_and_tree, __)
treebank_chords = unique(parse_chord(chord) 
  for tune in treebank for chord in leaflabels(tune.tree))

alterations = unique(alteration(c.root) for c in treebank_chords)
@assert all(a -> -1 <= a <= 1, alterations)

all_chords = collect(Chord(parsespelledpitch(letter * acc), form) 
  for letter in 'A':'G'
  for acc in ("b", "#", "")
  for form in instances(ChordForm))

# Tonal Pitch-Class Chord
const TPCC = Chord{Pitch{SpelledIC}}

startsym = H.start_category(TPCC)
ts       = H.terminal_category.(all_chords)
nts      = H.nonterminal_category.(all_chords)

start_rules = Set(H.start_rule.(nts))
nonstart_rules = Set([
  H.termination_rule(TPCC);
  H.duplication_rule(TPCC);
  H.leftheaded_rule.(nts);
  H.rightheaded_rule.(nts)])

rules = union(start_rules, nonstart_rules)
@test isbitstype(eltype(typeof(rules)))

uniform_dist(xs) = Dict( x => log(1/length(xs)) for x in xs )
@test uniform_dist(start_rules) |> values .|> exp |> sum |> isapprox(1)

params = (
  start_logprobs = uniform_dist(start_rules),
  nonstart_logprobs = Dict( nt => uniform_dist(nonstart_rules) 
                            for nt in nts ) )

import Distributions: logpdf
function logpdf(g::H.Grammar, lhs, rule)
  if H.start ⊣ lhs && H.startrule ⊣ rule
    g.params.start_logprobs[rule]
  elseif H.nonterminal ⊣ lhs && !(H.startrule ⊣ rule)
    g.params.nonstart_logprobs[lhs][rule]
  else
    log(0)
  end
end

grammar = H.Grammar(
  rules,
  # nt -> H.Category(H.terminal, nt.ntlabel, nt.tlabel),
  # t -> [H.Category(H.nonterminal, t.ntlabel, t.tlabel)],
  params)

terminalss = collect([H.terminal_category(c)]
  for c in [Chord(p"C", MAJ7), Chord(p"G", DOM), Chord(p"C", MAJ7)])
terminalss = fill([H.terminal_category(Chord(p"C", MAJ7))], 100)
scoring = AG.WDS(grammar)
@time chart = AG.chartparse(grammar, scoring, terminalss)
@time AG.sample_derivations(scoring, chart[1,100][startsym], 1) .|> 
  (app -> app.rule.tag)

############################################################################
