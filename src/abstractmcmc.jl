using Setfield
using AbstractMCMC: AbstractMCMC

import LinearAlgebra: ×

"""
    getparams([model, ]state)

Get the parameters from the `state`.

Default implementation uses [`getparams_and_logprob`](@ref).
"""
getparams(state) = first(getparams_and_logprob(state))
getparams(model, state) = first(getparams_and_logprob(model, state))

"""
    getlogprob([model, ]state)

Get the log probability of the `state`.

Default implementation uses [`getparams_and_logprob`](@ref).
"""
getlogprob(state) = last(getparams_and_logprob(state))
getlogprob(model, state) = last(getparams_and_logprob(model, state))

"""
    getparams_and_logprob([model, ]state)

Return a vector of parameters from the `state`.

See also: [`setparams_and_logprob!!`](@ref).
"""
getparams_and_logprob(model, state) = getparams_and_logprob(state)

"""
    setparams_and_logprob!!([model, ]state, params)

Set the parameters in the state to `params`, possibly mutating if it makes sense.

See also: [`getparams_and_logprob`](@ref).
"""
setparams_and_logprob!!(model, state, params, logprob) = setparams_and_logprob!!(state, params, logprob)

"""
    state_from_state(model, state_source, state_target[, transition_source, transition_target])

Return a new state similar to `state_target` but updated from `state_source`, which could be
a different type of state.
"""
function state_from_state(model, state_source, state_target, transition_source, transition_target)
    return state_from_state(model, state_source, state_target)
end
function state_from_state(model, state_source, state_target)
    params, logp = getparams_and_logprob(model, state_source)
    return setparams_and_logprob!!(model, state_target, params, logp)
end

"""
    SequentialTransitions

A `SequentialTransitions` object is a container for a sequence of transitions.
"""
struct SequentialTransitions{A}
    transitions::A
end

# Since it's a _sequence_ of transitions, the parameters and logprobs are the ones of the
# last transition/state.
getparams_and_logprob(transitions::SequentialTransitions) = getparams_and_logprob(transitions.transitions[end])
function getparams_and_logprob(model, transitions::SequentialTransitions)
    return getparams_and_logprob(model, transitions.transitions[end])
end

function setparams_and_logprob!!(transitions::SequentialTransitions, params, logprob)
    return @set transitions.transitions[end] = setparams_and_logprob!!(transitions.transitions[end], params, logprob)
end
function setparams_and_logprob!!(model, transitions::SequentialTransitions, params, logprob)
    return @set transitions.transitions[end] = setparams_and_logprob!!(model, transitions.transitions[end], params, logprob)
end

"""
    SequentialStates

A `SequentialStates` object is a container for a sequence of states.
"""
struct SequentialStates{A}
    states::A
end

# Since it's a _sequence_ of transitions, the parameters and logprobs are the ones of the
# last transition/state.
getparams_and_logprob(state::SequentialStates) = getparams_and_logprob(state.states[end])
getparams_and_logprob(model, state::SequentialStates) = getparams_and_logprob(model, state.states[end])

function setparams_and_logprob!!(state::SequentialStates, params, logprob)
    return @set state.states[end] = setparams_and_logprob!!(state.states[end], params, logprob)
end
function setparams_and_logprob!!(model, state::SequentialStates, params, logprob)
    return @set state.states[end] = setparams_and_logprob!!(model, state.states[end], params, logprob)
end

# Includes.
include("samplers/composition.jl")
include("samplers/repeated.jl")
include("samplers/multi.jl")

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

