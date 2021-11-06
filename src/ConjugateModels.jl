module ConjugateModels

# complete imports
using Distributions
using MacroTools

# partial imports
using SpecialFunctions: logbeta

# imports for overloading
import Base: rand, minimum, maximum, eltype
import Distributions: sampler, logpdf, support, insupport
import StatsBase: params

export DirCat, ConjugateModel, add_obs!, logpdf
export ConjugateModel, @conjugate_model

######################################
### Interface for conjugate models ###
######################################

# abbreviation: pscount ... pseudocount
function add_obs!(model, obs, pscount) end

# additionally: functions rand and logpdf

###########################################
### Dirichlet categorical distributions ###
###########################################

struct DirCat{T} <: Distribution{Univariate, Discrete}
  pscounts :: Dict{T, Float64}

  function DirCat(pscounts)
    @assert(
      !isempty(pscounts) && all(((x, pscount),) -> pscount > 0, pscounts), 
      "DirCat parameter invalid")
    new{keytype(pscounts)}(pscounts)
  end
end

eltype(::Type{DirCat{T}}) where T = T
params(dc::DirCat) = dc.pscounts
sampler(dc::DirCat) = dc
support(dc::DirCat) = collect(keys(dc.pscounts))
insupport(dc::DirCat{T}, x::T) where T = haskey(dc.pscounts, x)

function rand(rng::Distributions.AbstractRNG, dc::DirCat)
  probs = rand(rng, Dirichlet(collect(values(dc.pscounts))))
  k = rand(rng, Categorical(probs))
  collect(keys(dc.pscounts))[k]
end

function logpdf(dc::DirCat{T}, x::T) where T
  if haskey(dc.pscounts, x)
    logbeta(sum(values(dc.pscounts)), 1) - logbeta(dc.pscounts[x], 1)
  else
    -Inf
  end
end

function add_obs!(dc::DirCat{T}, x::T, pscount) where T
  @assert dc.pscounts[x] + pscount > 0 "DirCat parameter update invalid"
  dc.pscounts[x] += pscount
  dc
end

########################
### Conjugate models ###
########################

abstract type ConjugateModel end

# Generated functions don't work because they don't see method definitions from
# other modules. Therefore, we use a macro.

macro conjugate_model(model_expr)
  @capture(model_expr, struct (M_{Vars__} | M_) <: ConjugateModel fields__ end)
  variables = [f.args[1] for f in fields]

  # generate code for logpdf function
  logpdf_exprs = [:(logpdf($v(model, trace), trace.$v)) for v in variables]
  logpdf_fn_expr = quote
    function logpdf(model::$M, trace)
      +($(logpdf_exprs...))
    end
  end

  # generate code for rand function
  init_trace_expr = :(trace0 = @NamedTuple{}(()))
  sample_exprs = map(enumerate(variables)) do (i, v)
    :($(Symbol(:trace, i)) = 
        ($(Symbol(:trace, i-1))..., 
        $v=rand(rng, $v(model, $(Symbol(:trace, i-1))))))
  end
  rand_fn_expr = quote
    function rand(rng::Distributions.AbstractRNG, model::$M) 
      $(Expr(
        :block, 
        init_trace_expr, 
        sample_exprs..., 
        Symbol(:trace, length(variables))))
    end
  end

  # generate code for add_obs! function
  add_exprs = map(variables) do v 
    :(add_obs!($v(model, trace), trace.$v, pscount))
  end
  add_obs_fn_expr = quote 
    function add_obs!(model::$M, trace, pscount)
      $(Expr(:block, add_exprs..., :model))
    end
  end
  
  esc(
    :($(Expr(
      :block, 
      :(import Distributions: logpdf),
      :(import Base: rand),
      :(import AbstractGrammars.ConjugateModels: add_obs!),
      model_expr, 
      logpdf_fn_expr, 
      rand_fn_expr, 
      add_obs_fn_expr))))
end

# @generated function logpdf(model::M, trace) where M <: ConjugateModel
#   variables = fieldnames(M)
#   logpdf_exprs = [:(logpdf($v(model, trace), trace.$v)) for v in variables]
#   :(+($(logpdf_exprs...)))
# end

# @generated function rand(rng::Distributions.AbstractRNG, model::M) where 
#   M <: ConjugateModel

#   variables = fieldnames(M)
#   init_trace_expr = :(trace0 = @NamedTuple{}(()))
#   sample_exprs = map(enumerate(variables)) do (i, v)
#     :($(Symbol(:trace, i)) = ($(Symbol(:trace, i-1))..., 
#                               $v=rand(rng, $v(model, $(Symbol(:trace, i-1))))))
#   end
#   :($(Expr(:block, 
#     init_trace_expr, 
#     sample_exprs..., 
#     Symbol(:trace, length(variables)))))
# end

# @generated function add_obs!(model::M, trace, pscount) where M <: ConjugateModel
#   variables = fieldnames(M)
#   add_exprs = map(variables) do v 
#     :(add_obs!($v(model, trace), trace.$v, pscount))
#   end
#   :($(Expr(:block, add_exprs..., :model)))
# end

end # module

#################################################
### Pyro-style models using multiple dispatch ###
#################################################

# module PyroModels

# using Distributions
# using AbstractGrammars.ConjugateModels: DirCat

# abstract type ConjugateModel end

# struct Mixture{T} <: ConjugateModel
#   component :: DirCat{Int}
#   value     :: Vector{DirCat{T}}
# end

# function prob_prog(⊙, m::Mixture)
#   k = @sample(⊙, m.component)
#   @sample(⊙, m.value[k])
# end

# end # module

##########################################################################
### OLD CODE: Dirichlet categorical distributions on positive integers ###
##########################################################################

# struct DirichletCategorical{C <: Real, Cs <: AbstractVector{C}} <: Distribution{Univariate, Discrete}
#   pseudocounts :: Cs

#   function DirichletCategorical(pseudocounts::Cs) where {C, Cs <: AbstractVector{C}}
#     @assert(
#       length(pseudocounts) > 0 && all(c -> c > 0, pseudocounts),
#       "Parameter vector of Dirichlet Categorical distribution must be nonempty and positive.")
#     new{C, Cs}(pseudocounts)
#   end
# end

# params(dc::DirichletCategorical) = dc.pseudocounts
# sampler(dc::DirichletCategorical) = dc

# function rand(rng::Distributions.AbstractRNG, dc::DirichletCategorical)
#   probs = rand(rng, Dirichlet(dc.pseudocounts))
#   rand(rng, Categorical(probs))
# end

# logpdf(::DirichletCategorical, ::Real) = -Inf
# logpdf(dc::DirichletCategorical, x::Int) = 
#   if 1 <= x <= length(dc.pseudocounts)
#     logbeta(sum(dc.pseudocounts), 1) - logbeta(dc.pseudocounts[x], 1)
#   else
#     -Inf
#   end

# cdf(dc::DirichletCategorical, x::Real) = sum(exp(logpdf(dc, i)) for i in 0:ceil(Int, x))

# function quantile(dc::DirichletCategorical, q::Real)
#   i = 0
#   p = 0
#   while p < q
#     i += 1
#     p += exp(logpdf(dc, i))
#   end
#   i
# end

# minimum(dc::DirichletCategorical) = 1
# maximum(dc::DirichletCategorical) = length(dc.pseudocounts)

# add_obs!(dc::DirichletCategorical, obs::Int) = (dc.pseudocounts[obs] += 1; nothing)
# rm_obs!(dc::DirichletCategorical, obs::Int)  = (dc.pseudocounts[obs] -= 1; nothing)

# @testset "dirichlet categorical distribution interface" begin
#   # test constructor checks
#   @test_throws AssertionError DirichletCategorical([])
#   @test_throws AssertionError DirichletCategorical([-1])

#   # construct random distribution
#   n = rand(1:10)
#   pseudocounts = rand(1:10, n)
#   dc = DirichletCategorical(pseudocounts)
  
#   # check interface properties
#   @test params(dc) == pseudocounts
#   @test (minimum(dc), maximum(dc)) == (1, n)
#   @test support(dc) == 1:n
#   @test all(insupport(dc, x) for x in rand(dc, 100))
#   @test cdf(dc, maximum(dc)) ≈ 1
#   @test all(cdf(dc, quantile(dc, q) - 1) < q < cdf(dc, quantile(dc, q)) for q in rand(100))
#   @test all(quantile(dc, cdf(dc, x)) == x for x in rand(support(dc), 100))

#   # test observations
#   obs = rand(support(dc))
#   pseudocounts = deepcopy(dc.pseudocounts)
#   add_obs!(dc, obs)
#   @test pseudocounts != dc.pseudocounts
#   rm_obs!(dc, obs)
#   @test pseudocounts == dc.pseudocounts
# end
