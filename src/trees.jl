struct Tree{T}
  label :: T
  children :: Vector{Tree{T}}
end

Tree(label::T) where T = Tree(label, Tree{T}[])
Tree(label, children::Tree...) = Tree(label, collect(children))
isleaf(tree::Tree) = isempty(tree.children)
eltype(::Type{Tree{T}}) where T = T

function map(f, tree::Tree)
  if isleaf(tree)
    Tree(f(tree.label))
  else
    Tree(f(tree.label), [map(f, c) for c in tree.children])
  end
end

function dict2tree(f, dict; label_key="label", children_key="children")
  label = f(dict[label_key])
  T = typeof(label)
  children = Tree{T}[dict2tree(f, child) for child in dict[children_key]]
  Tree(label, children)
end

dict2tree(dict; args...) = dict2tree(identity, dict; args...)

function apply(derivation::Vector{<:AbstractRule}, c)
  i = 0 # rule index
  next_rule() = (i += 1; derivation[i])
  backtrack() = (i -= 1)
  are_there_more_rules() = (i < length(derivation))

  function rewrite(lhs)
    r = next_rule()
    rhs = apply(r, lhs)
    if isnothing(rhs) # rule r is not applicable to lhs
      backtrack()
      return Tree(lhs)
    else # rule r is applicable to lhs
      children = [are_there_more_rules() ? rewrite(c) : Tree(c) for c in rhs]
      return Tree(lhs, children)
    end
  end

  rewrite(c)
end

struct Treelet{T}
  root_label   :: T
  child_labels :: Vector{T}
end

function Treelet(root_label::T, child_labels::T...) where T
  Treelet(root_label, collect(child_labels))
end

arity(treelet::Treelet) = length(treelet.child_labels)

function treelets(tree::Tree{T}, out=Treelet{T}[]) where T
  treelet = Treelet(tree.label, map(child -> child.label, tree.children))
  push!(out, treelet)
  foreach(child -> treelets(child, out), tree.children)
  return out
end

labels(tree::Tree) = 
  [treelet.root_label for treelet in treelets(tree)]
leaflabels(tree::Tree) = 
  [treelet.root_label for treelet in treelets(tree) if arity(treelet) == 0]
innerlabels(tree::Tree) = 
  [treelet.root_label for treelet in treelets(tree) if arity(treelet) > 0]

tree2derivation(treelet2rule, tree::Tree) = 
  [treelet2rule(tl) for tl in treelets(tree) if arity(tl) >= 1]

function treelet2stdrule(treelet::Treelet)
  @assert arity(treelet) in (1, 2)
  StdRule(treelet.root_label, treelet.child_labels...)
end


function relabel_with_spans(tree)
  k = 0 # leaf index
  next_leafindex() = (k += 1; k)
  span(i, j) = (from=i, to=j)
  combine(span1, span2) = span(span1.from, span2.to)

  function relabel(tree) 
    if isleaf(tree)
      i = next_leafindex()
      Tree(span(i,i))
    elseif length(tree.children) == 1
      child = relabel(tree.children[1])
      Tree(child.label, child)
    elseif length(tree.children) == 2
      left  = relabel(tree.children[1])
      right = relabel(tree.children[2])
      Tree(combine(left.label, right.label), left, right)
    else
      error("tree is not binary")
    end
  end

  return relabel(tree)
end

function collapse_unaries(tree)
  if isleaf(tree)
    tree
  elseif length(tree.children) == 1
    collapse_unaries(tree.children[1])
  else
    Tree(tree.label, map(collapse_unaries, tree.children))
  end
end

function constituent_spans(tree)
  tree |> collapse_unaries |> relabel_with_spans |> innerlabels
end

function tree_similarity(tree1, tree2)
  spans1 = constituent_spans(tree1)
  spans2 = constituent_spans(tree2)
  @assert length(spans1) == length(spans2)
  length(intersect(spans1, spans2)) / length(spans1)
end
