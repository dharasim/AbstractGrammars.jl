# # Grammar Toolkit Tutorial
# ## Part 1: Simple Supervised Treebank Grammar
# Load treebank and plot a tree to get started
import AbstractGrammars.JazzTreebank as JHT
using AbstractGrammars: plot_tree

tunes, treebank = JHT.load_tunes_and_treebank(); length(treebank)
sunny = treebank[30];
getlabel = t -> replace(string(t.label.val), "♭"=>"b", "♯"=>"#")
plot_tree(sunny["harmony_tree"]; getlabel)

# Enumerate chord symbols
using Pitches: Pitches
all_chords = collect(
  JHT.Chord(Pitches.parsespelledpitch(letter * acc), form) 
  for letter in 'A':'G'
  for acc in ("b", "#", "")
  for form in instances(JHT.ChordForm))








1+1