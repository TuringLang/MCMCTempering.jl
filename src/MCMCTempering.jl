module MCMCTempering

import AbstractMCMC
import Distributions
import Random

using LogDensityProblems: LogDensityProblems
using ProgressLogging: ProgressLogging
using ConcreteStructs: @concrete
using Setfield: @set, @set!

using MCMCChains: MCMCChains

using InverseFunctions

using DocStringExtensions

include("logdensityproblems.jl")
include("abstractmcmc.jl")
include("adaptation.jl")
include("swapping.jl")
include("state.jl")
include("sampler.jl")
include("sampling.jl")
include("ladders.jl")
include("stepping.jl")
include("model.jl")

export tempered,
    tempered_sample,
    TemperedSampler,
    make_tempered_model,
    ReversibleSwap,
    NonReversibleSwap,
    SingleSwap,
    SingleRandomSwap,
    RandomSwap,
    NoSwap

# TODO: Should we make this trait-based instead?
implements_logdensity(x) = LogDensityProblems.capabilities(x) !== nothing
maybe_wrap_model(model) = implements_logdensity(model) ? AbstractMCMC.LogDensityModel(model) : model
maybe_wrap_model(model::AbstractMCMC.LogDensityModel) = model

end
