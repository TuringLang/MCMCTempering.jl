using Setfield
using AbstractMCMC: AbstractMCMC

import LinearAlgebra: ×

"""
    getparams([model, ]state)

Get the parameters from the `state`.

Default implementation uses [`getparams_and_logprob`](@ref).
"""
getparams(model, state) = first(getparams_and_logprob(model, state))

"""
    getlogprob([model, ]state)

Get the log probability of the `state`.

Default implementation uses [`getparams_and_logprob`](@ref).
"""
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

"""
    SequentialStates

A `SequentialStates` object is a container for a sequence of states.
"""
struct SequentialStates{A}
    states::A
end

# Since it's a _sequence_ of transitions, the parameters and logprobs are the ones of the
# last transition/state.
getparams_and_logprob(model, state::SequentialStates) = getparams_and_logprob(model, state.states[end])
function setparams_and_logprob!!(model, state::SequentialStates, params, logprob)
    return @set state.states[end] = setparams_and_logprob!!(model, state.states[end], params, logprob)
end

# We want to save all the transitions/states, so we need to append the new one.
function AbstractMCMC.save!!(
    samples::Vector,
    sample::SequentialTransitions,
    iteration::Integer,
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer=0;  # TODO: Dont' do this.
    kwargs...
)
    # NOTE: It's possible that `iteration + i > N`; can this cause issues? How do we deal with this?
    for (i, transition) in enumerate(sample.transitions)
        samples = AbstractMCMC.save!!(samples, transition, iteration + i, model, sampler, N; kwargs...)
    end
    return samples
end

# NOTE: When using a `SequentialTransition` we're not going to store that, but instead
# we're going to store whatever type of transitions it contains. Hence we should
# infer the type of the transitions from the type of the states.
function AbstractMCMC.samples(
    sample::SequentialTransitions,
    model::AbstractMCMC.AbstractModel,
    spl::AbstractMCMC.AbstractSampler,
    N::Integer;
    kwargs...
)
    return AbstractMCMC.samples(last(sample.transitions), model, spl, N; kwargs...)
end

function AbstractMCMC.samples(
    sample::SequentialTransitions,
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler;
    kwargs...
)
    return AbstractMCMC.samples(last(sample.transitions), model, sampler; kwargs...)
end

# Includes.
include("samplers/composition.jl")
include("samplers/repeated.jl")
include("samplers/multi.jl")
