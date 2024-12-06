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
    state_from(model_source, state_target, state_source)
    state_from(model_source, model_target, state_target, state_source)

Return a new state similar to `state_target` but updated from `state_source`, which could be
a different type of state.
"""
function state_from(model_target, model_source, state_target, state_source)
    return state_from(model_target, state_target, state_source)
end
function state_from(model_target, state_target, state_source)
    params, logp = getparams_and_logprob(state_source)
    return setparams_and_logprob!!(model_target, state_target, params, logp)
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
include("samplers/iid.jl")
