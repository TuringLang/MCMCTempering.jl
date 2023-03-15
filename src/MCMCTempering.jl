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
include("swapsampler.jl")
include("sampler.jl")
include("sampling.jl")
include("ladders.jl")
include("stepping.jl")
include("model.jl")
include("utils.jl")

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
# Bundling of non-tempered samples.
function bundle_nontempered_samples(
    ts::AbstractVector{<:TemperedTransition{<:SwapTransition,<:MultipleTransitions}},
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedSampler,
    state::TemperedState,
    ::Type{T};
    kwargs...
) where {T}
    # Create the same model and sampler as we do in the initial step for `TemperedSampler`.
    multimodel = MultiModel([
        make_tempered_model(sampler, model, sampler.chain_to_beta[i])
        for i in 1:numtemps(sampler)
    ])
    multisampler = MultiSampler([getsampler(sampler, i) for i in 1:numtemps(sampler)])
    multitransitions = [
        MultipleTransitions(sort_by_chain(ProcessOrder(), t.swaptransition, t.transition.transitions))
        for t in ts
    ]

    return AbstractMCMC.bundle_samples(
        multitransitions,
        multimodel,
        multisampler,
        MultipleStates(sort_by_chain(ProcessOrder(), state.swapstate, state.state.states)),
        T;
        kwargs...
    )
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:MultipleTransitions},
    model::MultiModel,
    sampler::MultiSampler,
    state::MultipleStates,
    # TODO: Generalize for any eltype `T`? Then need to overload for `Real`, etc.?
    ::Type{Vector{MCMCChains.Chains}};
    kwargs...
)
    return map(1:length(model), model.models, sampler.samplers, state.states) do i, model, sampler, state
        AbstractMCMC.bundle_samples([t.transitions[i] for t in ts], model, sampler, state, MCMCChains.Chains; kwargs...)
    end
end

# HACK: https://github.com/TuringLang/AbstractMCMC.jl/issues/118
function AbstractMCMC.bundle_samples(
    ts::Vector{<:TemperedTransition{<:SwapTransition,<:MultipleTransitions}},
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedSampler,
    state::TemperedState,
    ::Type{Vector{T}};
    bundle_resolve_swaps::Bool=false,
    kwargs...
) where {T}
    # TODO: Implement special one for `Vector{MCMCChains.Chains}`.
    if bundle_resolve_swaps
        return bundle_nontempered_samples(ts, model, sampler, state, Vector{T}; kwargs...)
    end

    # TODO: Do better?
    return ts
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:TemperedTransition{<:SwapTransition,<:MultipleTransitions}},
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedSampler,
    state::TemperedState,
    ::Type{Vector{MCMCChains.Chains}};
    kwargs...
)
    return bundle_nontempered_samples(ts, model, sampler, state, Vector{MCMCChains.Chains}; kwargs...)
end

function AbstractMCMC.bundle_samples(
    ts::AbstractVector{<:TemperedTransition{<:SwapTransition,<:MultipleTransitions}},
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedSampler,
    state::TemperedState,
    ::Type{MCMCChains.Chains};
    kwargs...
)
    # Extract the transitions ordered, which are ordered according to processes, according to the chains.
    ts_actual = [t.transition.transitions[first(t.swaptransition.chain_to_process)] for t in ts]
    return AbstractMCMC.bundle_samples(
        ts_actual,
        model,
        sampler_for_chain(sampler, state, 1),
        state_for_chain(state, 1),
        MCMCChains.Chains;
        kwargs...
    )
end

function AbstractMCMC.bundle_samples(
    ts::AbstractVector,
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler,
    state::CompositionState,
    ::Type{T};
    kwargs...
) where {T}
    # In the case of `!saveall(sampler)`, the state is not a `CompositionTransition` so we just propagate
    # the transitions to the `bundle_samples` for the outer stuff. Otherwise, we flatten the transitions.
    ts_actual = saveall(sampler) ? mapreduce(t -> [inner_transition(t), outer_transition(t)], vcat, ts) : ts
    # TODO: Should we really always default to outer sampler?
    return AbstractMCMC.bundle_samples(
        ts_actual, model, sampler.sampler_outer, state.state_outer, T;
        kwargs...
    )
end

# HACK: https://github.com/TuringLang/AbstractMCMC.jl/issues/118
function AbstractMCMC.bundle_samples(
    ts::Vector,
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler,
    state::CompositionState,
    ::Type{Vector{T}};
    kwargs...
) where {T}
    if !saveall(sampler)
        # In this case, we just use the `outer` for everything since this is the only
        # transitions we're keeping around.
        return AbstractMCMC.bundle_samples(
            ts, model, sampler.sampler_outer, state.state_outer, Vector{T};
            kwargs...
        )
    end

    # Otherwise, we don't know what to do.
    return ts
end

function AbstractMCMC.bundle_samples(
    ts::AbstractVector{<:CompositionTransition{<:MultipleTransitions,<:SwapTransition}},
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler{<:MultiSampler,<:SwapSampler},
    state::CompositionState{<:MultipleStates,<:SwapState},
    ::Type{T};
    bundle_resolve_swaps::Bool=false,
    kwargs...
) where {T}
    !bundle_resolve_swaps && return ts

    # Resolve the swaps.
    sampler_without_saveall = @set sampler.sampler_inner.saveall = Val(false)
    ts_actual = map(ts) do t
        composition_transition(sampler_without_saveall, inner_transition(t), outer_transition(t))
    end

    AbstractMCMC.bundle_samples(
        ts_actual, model, sampler.sampler_outer, state.state_outer, T;
        kwargs...
    )
end

# HACK: https://github.com/TuringLang/AbstractMCMC.jl/issues/118
function AbstractMCMC.bundle_samples(
    ts::Vector{<:CompositionTransition{<:MultipleTransitions,<:SwapTransition}},
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler{<:MultiSampler,<:SwapSampler},
    state::CompositionState{<:MultipleStates,<:SwapState},
    ::Type{Vector{T}};
    bundle_resolve_swaps::Bool=false,
    kwargs...
) where {T}
    !bundle_resolve_swaps && return ts

    # Resolve the swaps (using the already implemented resolution in `composition_transition`
    # for this particular sampler but without `saveall`).
    sampler_without_saveall = @set sampler.saveall = Val(false)
    ts_actual = map(ts) do t
        composition_transition(sampler_without_saveall, inner_transition(t), outer_transition(t))
    end

    return AbstractMCMC.bundle_samples(
        ts_actual, model, sampler.sampler_outer, state.state_outer, Vector{T};
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
