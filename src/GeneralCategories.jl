module GeneralCategories

export Category, start_category, terminal_category, nonterminal_category

using ..AbstractGrammars

import ..AbstractGrammars: default

# possible tags: start, terminal, nonterminal, default
struct Category{T}
  tag :: Tag
  val :: T

  function Category(tag, val::T) where T 
    @assert tag in ("terminal", "nonterminal")
    new{T}(tag, val)
  end
  function Category(tag, T::Type) 
    @assert tag in ("start", "default")
    new{T}(tag)
  end
end

default(::Type{Category{T}}) where T = Category("default", T)

start_category(T::Type) = Category("start", T)
terminal_category(val) = Category("terminal", val)
nonterminal_category(val) = Category("nonterminal", val)

terminal_category(c::Category) = Category("terminal", c.val)
nonterminal_category(c::Category) = Category("nonterminal", c.val)

end # module