using Memoize: @memoize

function calkin_wilf_children(x)
  a = numerator(x)
  b = denominator(x)
  return [a // (a+b), (a+b) // b]
end

@memoize function ratios_of_calkin_wilf_level(i)
  if i == 0
    [1//1]
  else
    mapreduce(calkin_wilf_children,append!, ratios_of_calkin_wilf_level(i-1))
  end
end

@memoize function proper_ratios_of_calkin_wilf_level(i)
  Set(filter(x->x<1, ratios_of_calkin_wilf_level(i)))
end

calkin_wilf_level(x::Rational) = stern_brocot_level(x)

# algorithm from https://en.wikipedia.org/wiki/Stern%E2%80%93Brocot_tree
function stern_brocot_path(x::Rational)
  @assert 0 < x
  path = Bool[]
  l = 0//1 # lower bound
  h = 1//0 # higher bound
  while true
    m = (numerator(l) + numerator(h)) // (denominator(l) + denominator(h))
    if x < m
      push!(path, false)
      h = m
    elseif x > m
      push!(path, true)
      l = m
    else
      break
    end
  end
  return path
end

function stern_brocot_level(x::Rational)
  @assert 0 < x
  level = 0
  l = 0//1 # lower bound
  h = 1//0 # higher bound
  while true
    m = (numerator(l) + numerator(h)) // (denominator(l) + denominator(h))
    if x < m
      level += 1
      h = m
    elseif x > m
      level += 1
      l = m
    else
      break
    end
  end
  return level
end

function calkin_wilf_sequence(max_level)
  n = sum(2^l for l in 0:max_level)
  seq = zeros(Rational{Int}, n)
  seq[1] = 1
  for i in 2:n
    seq[i] = 1 // (2*floor(seq[i-1]) - seq[i-1] + 1)
  end
  return seq
end