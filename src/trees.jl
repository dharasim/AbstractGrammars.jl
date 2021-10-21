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