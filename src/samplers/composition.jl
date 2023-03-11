"""
    CompositionSampler <: AbstractMCMC.AbstractSampler

A `CompositionSampler` is a container for a sequence of samplers.

# Fields
$(FIELDS)

# Examples
```julia
composed_sampler = sampler_inner ∘ sampler_outer # or `CompositionSampler(sampler_inner, sampler_outer, Val(true))`
AbstractMCMC.step(rng, model, composed_sampler) # one step of `sampler_inner`, and one step of `sampler_outer`
```
"""
struct CompositionSampler{S1,S2,SaveAll} <: AbstractMCMC.AbstractSampler
    "The outer sampler"
    sampler_outer::S1
    "The inner sampler"
    sampler_inner::S2
    "Whether to save all the transitions or just the last one"
    saveall::SaveAll
end

CompositionSampler(sampler_outer, sampler_inner) = CompositionSampler(sampler_outer, sampler_inner, Val(true))

Base.:∘(s_outer::AbstractMCMC.AbstractSampler, s_inner::AbstractMCMC.AbstractSampler) = CompositionSampler(s_outer, s_inner)

"""
    saveall(sampler)

Return whether the sampler saves all the transitions or just the last one.
"""
saveall(sampler::CompositionSampler) = sampler.saveall
saveall(::CompositionSampler{<:Any,<:Any,Val{SaveAll}}) where {SaveAll} = SaveAll

"""
    CompositionState

Wrapper around the inner and outer states obtained from a [`CompositionSampler`](@ref).

# Fields
$(FIELDS)
"""
struct CompositionState{S1,S2}
    "The outer state"
    state_outer::S1
    "The inner state"
    state_inner::S2
end

getparams_and_logprob(state::CompositionState) = getparams_and_logprob(state.state_outer)
getparams_and_logprob(model, state::CompositionState) = getparams_and_logprob(model, state.state_outer)

function setparams_and_logprob!!(state::CompositionState, params, logprob)
    return @set state.state_outer = setparams_and_logprob!!(state.state_outer, params, logprob)
end
function setparams_and_logprob!!(model, state::CompositionState, params, logprob)
    return @set state.state_outer = setparams_and_logprob!!(model, state.state_outer, params, logprob)
end

"""
    CompositionTransition

Wrapper around the inner and outer transitions obtained from a [`CompositionSampler`](@ref).

# Fields
$(FIELDS)
"""
struct CompositionTransition{S1,S2}
    "The outer transition"
    transition_outer::S1
    "The inner transition"
    transition_inner::S2
end

# Useful functions for interacting with composition sampler and states.
inner_sampler(sampler::CompositionSampler) = sampler.sampler_inner
outer_sampler(sampler::CompositionSampler) = sampler.sampler_outer

inner_state(state::CompositionState) = state.state_inner
outer_state(state::CompositionState) = state.state_outer

inner_transition(transition::CompositionTransition) = transition.transition_inner
outer_transition(transition::CompositionTransition) = transition.transition_outer
outer_transition(transition) = transition  # in case we don't have `saveall`

# TODO: We really don't need to use `SequentialStates` here, do we?
composition_state(sampler, state_inner, state_outer) = CompositionState(state_outer, state_inner)
function composition_transition(sampler, transition_inner, transition_outer)
    return if saveall(sampler)
        CompositionTransition(transition_outer, transition_inner)
    else
        transition_outer
    end
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler;
    kwargs...
)
    state_inner_initial = last(AbstractMCMC.step(rng, model, inner_sampler(sampler); kwargs...))
    state_outer_initial = last(AbstractMCMC.step(rng, model, outer_sampler(sampler); kwargs...))

    # Create the composition state, and take a full step.
    state = composition_state(sampler, state_inner_initial, state_outer_initial)
    return AbstractMCMC.step(rng, model, sampler, state; kwargs...)
end

# TODO: Do we even need two versions? We could technically use `SequentialStates`
# in place of `CompositionState` and just have one version.
# The annoying part here is that we'll have to check `saveall` on every `step`
# rather than just for the initial step.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler,
    state;
    kwargs...
)
    state_inner_prev, state_outer_prev = inner_state(state), outer_state(state)

    # Update the inner state.
    current_state_inner = state_from(model, state_inner_prev, state_outer_prev)

    # Take a step in the inner sampler.
    transition_inner, state_inner = AbstractMCMC.step(rng, model, sampler.sampler_inner, current_state_inner; kwargs...)

    # Take a step in the outer sampler.
    current_state_outer = state_from(model, state_outer_prev, state_inner)
    transition_outer, state_outer = AbstractMCMC.step(rng, model, sampler.sampler_outer, current_state_outer; kwargs...)

    return (
        composition_transition(sampler, transition_inner, transition_outer),
        composition_state(sampler, state_inner, state_outer)
    )
end
