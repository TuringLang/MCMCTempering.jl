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

A `CompositionState` is a container for a sequence of states.

# Fields
$(FIELDS)
"""
struct CompositionState{S1,S2}
    "The outer state"
    state_outer::S1
    "The inner state"
    state_inner::S2
end

getparams_and_logprob(model, state::CompositionState) = getparams_and_logprob(model, state.state_outer)
function setparams_and_logprob!!(model, state::CompositionState, params, logprob)
    return @set state.state_outer = setparams_and_logprob!!(model, state.state_outer, params, logprob)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler;
    kwargs...
)
    state_inner_initial = last(AbstractMCMC.step(rng, model, sampler.sampler_inner; kwargs...))
    state_outer_initial = last(AbstractMCMC.step(rng, model, sampler.sampler_outer; kwargs...))

    # Create the composition state, and take a full step.
    state = if saveall(sampler)
        SequentialStates((state_inner_initial, state_outer_initial))
    else
        CompositionState(state_outer_initial, state_inner_initial)
    end
    return AbstractMCMC.step(rng, model, sampler, state; kwargs...)
end

# TODO: Do we even need two versions? We could technically use `SequentialStates`
# in place of `CompositionState` and just have one version.
# The annoying part here is that we'll have to check `saveall` on every `step`
# rather than just for the initial step.

# NOTE: Version which does keep track of all transitions and states.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler,
    state::SequentialStates;
    kwargs...
)
    @assert length(state.states) == 2 "Composition samplers only support MultipleStates with two states."

    state_inner_prev, state_outer_prev = state.states

    # Update the inner state.
    current_state_inner = state_from_state(model, state_outer_prev, state_inner_prev)

    # Take a step in the inner sampler.
    transition_inner, state_inner = AbstractMCMC.step(rng, model, sampler.sampler_inner, current_state_inner; kwargs...)

    # Take a step in the outer sampler.
    current_state_outer = state_from_state(model, state_inner, state_outer_prev)
    transition_outer, state_outer = AbstractMCMC.step(rng, model, sampler.sampler_outer, current_state_outer; kwargs...)

    return SequentialTransitions((transition_inner, transition_outer)), SequentialStates((state_inner, state_outer))
end

# NOTE: Version which does NOT keep track of all transitions and states.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::CompositionSampler,
    state::CompositionState;
    kwargs...
)
    # Update the inner state.
    current_state_inner = state_from_state(model, state.state_outer, state.state_inner)

    # Take a step in the inner sampler.
    state_inner = last(AbstractMCMC.step(rng, model, sampler.sampler_inner, current_state_inner; kwargs...))

    # Take a step in the outer sampler.
    current_state_outer = state_from_state(model, state_inner, state.state_outer)
    transition_outer, state_outer = AbstractMCMC.step(rng, model, sampler.sampler_outer, current_state_outer; kwargs...)

    # Create the composition state.
    state = CompositionState(state_outer, state_inner)

    return transition_outer, state
end
