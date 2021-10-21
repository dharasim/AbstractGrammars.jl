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

#######################################
### Short lists of limited max size ###
#######################################

module AtMosts

import Base: length, getindex, iterate, eltype, show

export AtMost, AtMost0, AtMost1, AtMost2, AtMost3, AtMost4

abstract type AtMost{T} end

length(xs::AtMost) = xs.length.val
getindex(xs::AtMost, i) = getfield(xs, i+1)
eltype(::Type{A}) where {T, A <: AtMost{T}} = T

function iterate(xs::AtMost, i=0)
  if i == length(xs)
    nothing
  else
    (xs[i+1], i+1)
  end
end

function show(io::IO, iter::AtMost{T}) where T
  print(io, "AtMost{$T}(")
  for x in iter
    print(io, x, ",")
  end
  print(io, ")")
end

struct Length val::Int end

function generate_code_atmost(n)
  struct_expr = quote
    struct $(Symbol(:AtMost, n)){T} <: AtMost{T}
      length::Length
    end
  end
  append!( # append field expressions
    struct_expr.args[2].args[3].args, 
    [:($(Symbol(:val, i)) :: T) for i in 1:n])
  
  struct_args = struct_expr.args[2].args[3].args
  ex = quote
    function $(Symbol(:AtMost, n))(T::Type)
      new{T}(Length(0))
    end
  end
  push!(struct_args, ex.args[2])
  for k in 1:n
    vals = Symbol.(:val, 1:k)
    typed_vals = Expr.(:(::), vals, :T)
    ex = quote
      function $(Symbol(:AtMost, n))($(typed_vals...),) where T
        new{T}(Length($k), $(vals...))
      end
    end
    push!(struct_args, ex.args[2])
  end

  # ex = Expr(:block, struct_expr, constr_exprs)
  eval(struct_expr)
end

for n in 0:4
  generate_code_atmost(n)
end

end # module

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
   tree = @_ tune["trees"][1]["open_constituent_tree"] |> 
             dict2tree(remove_asterisk, __) |>
             map(parse_chord, __))
end

function innerlabels(tree::Tree{L}) where L
  labels = L[]
  pushlabels(::Leaf) = nothing
  function pushlabels(tree::Binary)
    push!(labels, tree.label)
    pushlabels(tree.left)
    pushlabels(tree.right)
  end

  pushlabels(tree)
  return labels
end

function leaflabels(tree::Tree{L}) where L
  labels = L[]
  pushlabels(tree::Leaf) = push!(labels, tree.label)
  pushlabels(tree::Binary) = (pushlabels(tree.left); pushlabels(tree.right))
  
  pushlabels(tree)
  return labels
end

function relabel_with_spans(tree)
  k = 0 # leaf index
  next_leafindex() = (k += 1; k)
  span(i, j) = (from=i, to=j)
  combine(span1, span2) = span(span1.from, span2.to)

  function relabel(tree::Leaf) 
    i=next_leafindex()
    Leaf(span(i,i))
  end
  function relabel(tree::Binary) 
    left = relabel(tree.left)
    right = relabel(tree.right)
    Binary(combine(left.label, right.label), left, right)
  end

  return relabel(tree)
end

constituent_spans(tree) = tree |> relabel_with_spans |> innerlabels

function tree_similarity(tree1, tree2)
  spans1 = constituent_spans(tree1)
  spans2 = constituent_spans(tree2)
  @assert length(spans1) == length(spans2)
  length(intersect(spans1, spans2)) / length(spans1)
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

##################################
### General context-free rules ###
##################################

using .AtMosts

struct CFRule{C} <: AG.AbstractRule{C}
  lhs :: C
  rhs :: AtMost2{C}
end

CFRule(lhs, rhs...) = CFRule(lhs, AtMost2(rhs...))
arity(r::CFRule) = length(r.rhs)
# @assert [CFRule('a', 'b', 'c'), CFRule('d', 'e')] |> eltype |> isbitstype

function derivation(tree::Tree{T}) where T
  rules = [CFRule(H.start_cat(T), H.nonterminal_cat(tree.label))]

  push_rules(tree::Leaf) = begin 
    r = CFRule(H.nonterminal_cat(tree.label), H.terminal_cat(tree.label))
    push!(rules, r)
  end
  push_rules(tree::Binary) = begin
    r = CFRule(
      H.nonterminal_cat(tree.label), 
      H.nonterminal_cat(tree.left.label), 
      H.nonterminal_cat(tree.right.label))
    push!(rules, r)
    push_rules(tree.left)
    push_rules(tree.right)
  end

  push_rules(tree)
  return rules
end

function derivation2tree(grammar, derivation::Vector{AG.App{C, R}}) where {C, R}
  i = 0 # rule index
  next_app() = (i += 1; derivation[i])
  
  function rewrite(nt)
    app = next_app()
    @assert nt == app.lhs
    rhs = AG.apply(grammar, app)
    if length(rhs) == 1 # terminal rule
      Leaf(rhs[1])
    elseif length(rhs) == 2 # binary rule
      Binary(nt, rewrite(rhs[1]), rewrite(rhs[2]))
    else
      error("only binary rules and unary termination rules are supported")
    end
  end

  app1, app2 = derivation[1:2]
  if length(AG.apply(grammar, app1)) == 1
    # if initial rule is unary, then skip it for the tree
    i += 1
    rewrite(app2.lhs)
  else
    rewrite(app1.lhs)
  end
end

function cfrule_to_headedrule_app(r::CFRule{H.Category{T}}) where T
    if arity(r) == 1 && H.start ⊣ r.lhs && H.nonterminal ⊣ r.rhs[1]
      return AG.App(r.lhs, H.start_rule(r.rhs[1]))
    elseif arity(r) == 1 && H.nonterminal ⊣ r.lhs && H.terminal ⊣ r.rhs[1]
      return AG.App(r.lhs, H.termination_rule(T))
    elseif arity(r) == 2 && H.nonterminal ⊣ (r.lhs, r.rhs...)
      if r.lhs == r.rhs[1] == r.rhs[2]
        return AG.App(r.lhs, H.duplication_rule(T))
      elseif r.lhs == r.rhs[1] && r.rhs[1] != r.rhs[2]
        return AG.App(r.lhs, H.leftheaded_rule(r.rhs[2]))
      elseif r.lhs == r.rhs[2] && r.rhs[1] != r.rhs[2]
        return AG.App(r.lhs, H.rightheaded_rule(r.rhs[1]))
      end
    end
    error("$r could not be converted into a headed rule")
end

############
### Main ###
############

treebank_url = "https://raw.githubusercontent.com/DCMLab/JazzHarmonyTreebank/master/treebank.json"
tunes = HTTP.get(treebank_url).body |> String |> JSON.parse
treebank = @_ tunes |> filter(haskey(_, "trees"), __) |> map(title_and_tree, __)
derivations = [derivation(tune.tree) for tune in treebank] 
rule_apps = [cfrule_to_headedrule_app(rule) for d in derivations for rule in d]

@assert begin 
  treebank_chords = unique(chord for tune in treebank for chord in leaflabels(tune.tree))
  alterations = unique(alteration(c.root) for c in treebank_chords) 
  all(a -> -1 <= a <= 1, alterations)
end

all_chords = collect(
  Chord(parsespelledpitch(letter * acc), form) 
  for letter in 'A':'G'
  for acc in ("b", "#", "")
  for form in instances(ChordForm))

# Tonal Pitch-Class Chord
const TPCC = Chord{Pitch{SpelledIC}}

startsym = H.start_cat(TPCC)
ts       = H.terminal_cat.(all_chords)
nts      = H.nonterminal_cat.(all_chords)

start_rules = Set(H.start_rule.(nts))
nonstart_rules = Set([
  H.termination_rule(TPCC);
  H.duplication_rule(TPCC);
  H.leftheaded_rule.(nts);
  H.rightheaded_rule.(nts)])

rules = union(start_rules, nonstart_rules)
@test isbitstype(eltype(typeof(rules)))

# uniform_dist(xs) = Dict( x => log(1/length(xs)) for x in xs )
# @test uniform_dist(start_rules) |> values .|> exp |> sum |> isapprox(1)

using AbstractGrammars.ConjugateModels: DirCat, add_obs!
flat_dircat(xs) = DirCat(Dict(x => 1 for x in xs))

params = (
  start_dist = flat_dircat(start_rules),
  nonstart_dists = Dict( nt => flat_dircat(nonstart_rules) 
                            for nt in nts ) )

import Distributions: logpdf
function logpdf(g::H.Grammar, lhs, rule)
  if H.start ⊣ lhs && H.startrule ⊣ rule
    logpdf(g.params.start_dist, rule)
  elseif H.nonterminal ⊣ lhs && !(H.startrule ⊣ rule)
    logpdf(g.params.nonstart_dists[lhs], rule)
  else
    log(0)
  end
end

function observe_tree!(params, tree)
  apps = cfrule_to_headedrule_app.(derivation(tree))
  for app in apps
    if H.startrule ⊣ app.rule
      add_obs!(params.start_dist, app.rule, 1)
    else
      add_obs!(params.nonstart_dists[app.lhs], app.rule, 1)
    end
  end
end

# observe treebank
foreach(tune -> observe_tree!(params, tune.tree), treebank)

grammar = H.Grammar(rules, params)

terminalss = collect([H.terminal_cat(c)]
  for c in [Chord(p"C", MAJ7), Chord(p"G", DOM), Chord(p"C", MAJ7)])
terminalss = fill([H.terminal_cat(Chord(p"C", MAJ7))], 100)

scoring = AG.WDS(grammar)
@time chart = AG.chartparse(grammar, scoring, terminalss)
@time AG.sample_derivations(scoring, chart[1,length(terminalss)][startsym], 1) .|> 
  (app -> app.rule.tag)

scoring = AG.BestDerivationScoring()
accs = zeros(length(treebank))
@time for i in eachindex(treebank)
  print(i, ' ', treebank[i].title, ' ')
  tree = treebank[i].tree
  terminalss = [[H.terminal_cat(c)] for c in leaflabels(tree)]
  chart = AG.chartparse(grammar, scoring, terminalss)
  apps = chart[1, length(terminalss)][startsym].apps
  accs[i] = tree_similarity(tree, derivation2tree(grammar, apps))
  println(accs[i])
end

sum(accs) / 150

############################################################################
