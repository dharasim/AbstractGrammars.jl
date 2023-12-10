# AbstractGrammars.jl

Work-in-progress implementation of flexible probabilistic grammars in Julia.

## Running the tutorial

* Install Julia, go to your clone
* `julia --project=.`
* package mode: `]`, then `instantiate` (installs dependencies)
  * problems with the Manifest? Try `update`
  * `add IJulia`
* `using Pkg`
* `ENV["IJULIA_DEBUG"]=true`
* `Pkg.build("AbstractGrammars")`
* `Pkg.build("IJulia")`
* `using IJulia`
* `installkernel("Julia Grammars", "--project=$(Base.active_project())")`
* `notebook(dir=".")`