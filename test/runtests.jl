using Test
using Distributions
using AbstractGrammars
using AbstractGrammars.ConjugateModels

###################################
### Test compound distributions ###
###################################

@testset "dirichlet categorical" begin
  dc = DirCat(Dict('a' => 5, 'b' => 3))
  @test exp(logpdf(dc, 'a')) > 0
  @test insupport(dc, 'a')
  @test add_obs!(dc, 'a', +10).pscounts['a'] == 15
end

###############################
### Example conjugate model ###
###############################

@conjugate_model struct Mixture{T} <: ConjugateModel
  component :: DirCat{Int}
  value     :: Vector{DirCat{T}}
end

# get the distributions selected by a trace
component(m::Mixture, trace) = m.component
value(m::Mixture, trace) = m.value[trace.component]

@testset "mixture model example" begin
  model = Mixture(
    DirCat(Dict(zip(1:3, rand(1:10, 3)))), 
    [DirCat(Dict(zip('a':'e', rand(1:10, 5)))) for _ in 1:3])

  trace = rand(model)
  @test exp(logpdf(model, trace)) > 0
  pscount = model.component.pscounts[trace.component]
  @test add_obs!(model, trace, 10).component.pscounts[trace.component] == 
        pscount + 10
end

###############################
### Test binary-tree counts ###
###############################

import AbstractGrammars.BinaryCountGrammar
@testset "count all binary trees" begin
  BinaryCountGrammar.test_binary_count_grammar()
end