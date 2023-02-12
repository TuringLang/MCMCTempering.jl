module MCMCTempering

import AbstractMCMC
import Distributions
import Random

using LogDensityProblems: LogDensityProblems
using ProgressLogging: ProgressLogging
using ConcreteStructs: @concrete
using Setfield: @set, @set!

using InverseFunctions

using DocStringExtensions

include("logdensityproblems.jl")
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
    StandardSwap,
    RandomPermutationSwap,
    NonReversibleSwap,
    NoSwap,
    maybe_wrap_model

# TODO: Should we make this trait-based instead?
implements_logdensity(x) = LogDensityProblems.capabilities(x) !== nothing
maybe_wrap_model(model) = implements_logdensity(model) ? AbstractMCMC.LogDensityModel(model) : model
maybe_wrap_model(model::AbstractMCMC.LogDensityModel) = model

function AbstractMCMC.bundle_samples(
    ts::AbstractVector,
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedSampler,
    state::TemperedState,
    chain_type::Type;
    kwargs...
)
    AbstractMCMC.bundle_samples(
        ts, maybe_wrap_model(model), get_sampler(sampler, state.chain_order[1]), get_state(state), chain_type;
        kwargs...
    )
end

end
