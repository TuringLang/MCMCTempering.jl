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
include("swapsampler.jl")
include("tempered_composition.jl")

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

# Bundling.
# TODO: Improve this, somehow.
# TODO: Move this to an extension.
function AbstractMCMC.bundle_samples(
    ts::AbstractVector{<:TemperedTransition},
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedSampler,
    state::TemperedState,
    ::Type{MCMCChains.Chains};
    kwargs...
)
    return AbstractMCMC.bundle_samples(
        map(Base.Fix2(getproperty, :transition), filter(!Base.Fix2(getproperty, :is_swap), ts)),  # Remove the swaps.
        model,
        sampler_for_chain(sampler, state),
        state_for_chain(state),
        MCMCChains.Chains;
        kwargs...
    )
end

function AbstractMCMC.bundle_samples(
    ts::AbstractVector,
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler,
    state::CompositionState,
    ::Type{MCMCChains.Chains};
    kwargs...
)
    return AbstractMCMC.bundle_samples(
        ts, model, sampler.sampler_outer, state.state_outer, MCMCChains.Chains;
        kwargs...
    )
end

# Unflatten in the case of `SequentialTransitions`
function AbstractMCMC.bundle_samples(
    ts::AbstractVector{<:SequentialTransitions},
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler,
    state::SequentialStates,
    ::Type{MCMCChains.Chains};
    kwargs...
)
    ts_actual = [t for tseq in ts for t in tseq.transitions]
    return AbstractMCMC.bundle_samples(
        ts_actual, model, sampler.sampler_outer, state.states[end], MCMCChains.Chains;
        kwargs...
    )
end

function AbstractMCMC.bundle_samples(
    ts::AbstractVector,
    model::AbstractMCMC.AbstractModel,
    sampler::RepeatedSampler,
    state,
    ::Type{MCMCChains.Chains};
    kwargs...
)
    return AbstractMCMC.bundle_samples(ts, model, sampler.sampler, state, MCMCChains.Chains; kwargs...)
end

# Unflatten in the case of `SequentialTransitions`.
function AbstractMCMC.bundle_samples(
    ts::AbstractVector{<:SequentialTransitions},
    model::AbstractMCMC.AbstractModel,
    sampler::RepeatedSampler,
    state::SequentialStates,
    ::Type{MCMCChains.Chains};
    kwargs...
)
    ts_actual = [t for tseq in ts for t in tseq.transitions]
    return AbstractMCMC.bundle_samples(
        ts_actual, model, sampler.sampler, state.states[end], MCMCChains.Chains;
        kwargs...
    )
end

end
