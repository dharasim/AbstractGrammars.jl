module JazzTreebank

export TPCC, ChordForm, parse_chord, load_tunes_and_treebank, chord_durations

# named imports
import HTTP, JSON

# imports for overloading
import Base: show
import ..AbstractGrammars: default

# imports without overloading
using ..AbstractGrammars: normalize, Tree, dict2tree, isleaf, leaflabels, zip_trees, T, NT
using Pitches: parsespelledpitch, Pitch, SpelledIC, MidiIC, midipc, alteration, @p_str, tomidi
using OffsetArrays: OffsetArray
using Underscores: @_

#############################################
### Defaults for pitch and interval types ###
#############################################

default(::Type{SpelledIC}) = SpelledIC(0)
default(::Type{Pitch{I}}) where I = Pitch(default(I))

##############
### Chords ###
##############

@enum ChordForm MAJ MAJ6 MAJ7 DOM MIN MIN6 MIN7 MINMAJ7 HDIM7 DIM7 SUS

const chordform_strings = OffsetArray( # zero indexed
  ["^", "6", "^7", "7", "m", "m6", "m7", "m^7", "%7", "o7", "sus"], -1)

chordform_string(form::ChordForm) = chordform_strings[Int(form)]

function parse_chordform(str::AbstractString)
  i = findfirst(isequal(str), chordform_strings)
  @assert !isnothing(i) "$str cannot be parsed as a chord form"
  return ChordForm(i)
end

default(::Type{ChordForm}) = ChordForm(0)

struct Chord{R}
  root :: R
  form :: ChordForm
end

# Tonal Pitch-Class Chord
const TPCC = Chord{Pitch{SpelledIC}}

show(io::IO, chord::Chord) = print(io, chord.root, chordform_string(chord.form))

function default(::Type{Chord{R}}) where R 
  Chord(default(R), default(ChordForm))
end

const chord_regex = r"([A-G]b*|[A-G]#*)([^A-Gb#]+)"

function parse_chord(str)
  m = match(chord_regex, str)
  @assert !isnothing(m) "$str cannot be parsed as a pitch-class chord"
  root_str, form_str = m.captures
  root = parsespelledpitch(root_str)
  form = parse_chordform(form_str)
  return Chord(root, form)
end

# tests
@assert all(instances(ChordForm)) do form
  form |> chordform_string |> parse_chordform == form
end
@assert default(Chord{Pitch{SpelledIC}}) == parse_chord("C^")

##############################
### Basic treebank reading ###
##############################

function categorize_and_insert_terminal_rules(tree)
  function categorize(tree)
    if isleaf(tree)
      Tree(NT(tree.label), Tree(T(tree.label)))
    else
      Tree(NT(tree.label), map(categorize, tree.children))
    end
  end
  categorize(tree)
end

function preprocess_tree!(tune)
  remove_asterisk(label::String) = replace(label, "*" => "")

  if haskey(tune, "trees")
    tune["harmony_tree"] = @_ tune["trees"][1]["open_constituent_tree"] |> 
      dict2tree(remove_asterisk, __) |>
      map(parse_chord, __) |>
      categorize_and_insert_terminal_rules(__)
  end

  return tune
end

##############################
### Rhythm data processing ###
##############################

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
  ls = leaflabels(tune["harmony_tree"])
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

const url = "https://raw.githubusercontent.com/DCMLab/JazzHarmonyTreebank/master/treebank.json"

function load_tunes_and_treebank()
  tunes = HTTP.get(url).body |> String |> JSON.parse .|> preprocess_tree!
  # tunes = open("data/treebank.json") do file
  #   read(file, String) |> JSON.parse .|> preprocess_tree!
  # end
  treebank = filter(tune -> haskey(tune, "harmony_tree"), tunes)
  for tune in treebank
    tune["rhythm_tree"] = categorize_and_insert_terminal_rules(
      normalized_duration_tree(tune))
    tune["product_tree"] = zip_trees(tune["harmony_tree"], tune["rhythm_tree"])
  end
  @assert count(tune->haskey(tune, "product_tree"), tunes) == length(treebank)
  return tunes, treebank
end

end # module