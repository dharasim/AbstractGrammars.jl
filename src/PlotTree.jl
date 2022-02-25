module PlotTree

using Compose
export plot_tree

function plot_tree(t; 
    getchildren=t->t.children, getlabel=t->t.label, 
    textcolor="black", linecolor="grey", white_background=true, save_as=nothing,
    scale_width=1, scale_heigth=1
  )
  
  # ad-hoc tree data type
  tree(label, nleafs, height, children=[]) = (; label, nleafs, height, children)
  leaf(label) = tree(label, 1, 1)
  merge(label, ts...) = tree(
    label,
    sum(t.nleafs for t in ts), 
    maximum(t.height for t in ts) + 1,
    collect(ts),
  )

  # recursive function converting any tree into the ad-hoc tree type
  function convert_tree(t)
    children = getchildren(t)
    label = getlabel(t)
    if isempty(children)
      leaf(label)
    else
      converted_children = map(c -> convert_tree(c), children)
      merge(label, converted_children...)
    end
  end

  # recursive function mapping ad-hoc trees to Compose.jl compositions
  function tree_composition(t)
    if isempty(t.children)
      compose(context(), text(0.5, 0.5, t.label, hcenter, vcenter), fill(textcolor))
    else
      # comps ... compositions
      child_comps = map(tree_composition, t.children)
      child_contexts = []
      lines = []
      x = 0
      for c in t.children
        y = 1 - c.height / t.height
        w = c.nleafs / t.nleafs
        h = c.height / t.height
        push!(child_contexts, context(x, y, w, h))
        push!(lines, line([(0.5, 1/(1.5*t.height)), (x + w/2, y + 1/(4*t.height))]))
        x += c.nleafs/t.nleafs
      end
      compose(context(), 
        compose(context(), lines..., stroke(linecolor)), 
        compose(
          context(0, 0, 1, 1/t.height), 
          text(0.5, 0.5, t.label, hcenter, vcenter), 
          fill(textcolor)
        ), 
        compose.(child_contexts, child_comps)...)
    end
  end

  background = if white_background 
    compose(context(), rectangle(), fill("white"))
  else
    compose(context())
  end

  ct = convert_tree(t)
  comp = compose(context(), tree_composition(ct), background)

  # measurements of the output picture
  width = scale_width * ct.nleafs * cm
  height = scale_heigth * ct.height * cm

  # save picture to disk
  if !isnothing(save_as)
    file_extension = splitext(save_as)[2]
    format = if file_extension == ".png"
      PNG
    elseif file_extension == ".svg"
      SVG
    elseif file_extension == ".pdf"
      PDF
    else
      error("supported tree-plot formats: .png .svg .pdf")
    end
    draw(format(save_as, width, height), comp)
  end

  # output picture as PNG
  comp |> SVG(width, height) 
end

end # module