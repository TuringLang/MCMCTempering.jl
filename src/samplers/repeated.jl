"""
    RepeatedSampler <: AbstractMCMC.AbstractSampler

A `RepeatedSampler` is a container for a sampler and a number of times to repeat it.

# Fields
$(FIELDS)

# Examples
```julia
repeated_sampler = sampler^10 # or `RepeatedSampler(sampler, 10, Val(true))`
AbstractMCMC.step(rng, model, repeated_sampler) # take 10 steps of `sampler`
```
"""
struct RepeatedSampler{S,SaveAll} <: AbstractMCMC.AbstractSampler
    "The sampler to repeat"
    sampler::S
    "The number of times to repeat the sampler"
    num_repeat::Int
    "Whether to save all the transitions or just the last one"
    saveall::SaveAll
end

RepeatedSampler(sampler, num_repeat) = RepeatedSampler(sampler, num_repeat, Val(true))

Base.@constprop :aggressive Base.:^(s::AbstractMCMC.AbstractSampler, n::Int) = RepeatedSampler(s, n, Val(true))

saveall(sampler::RepeatedSampler) = sampler.saveall
saveall(::RepeatedSampler{<:Any,Val{SaveAll}}) where {SaveAll} = SaveAll

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::RepeatedSampler;
    kwargs...
)
    transition, state = AbstractMCMC.step(rng, model, sampler.sampler; kwargs...)
    if saveall(sampler)
        return SequentialTransitions([transition]), SequentialStates([state])
    else
        return transition, state
    end
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::RepeatedSampler,
    state;
    kwargs...
)
    # Take a step in the inner sampler.
    transition, state = AbstractMCMC.step(rng, model, sampler.sampler, state; kwargs...)

    # Take a step in the outer sampler.
    for _ in 2:sampler.num_repeat
        transition, state = AbstractMCMC.step(rng, model, sampler.sampler, state; kwargs...)
    end

    return transition, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::RepeatedSampler,
    state::SequentialStates;
    kwargs...
)
    # Take a step in the inner sampler.
    transition, state_inner = AbstractMCMC.step(rng, model, sampler.sampler, state.states[end]; kwargs...)

    # Take a step in the outer sampler.
    transitions = [transition]
    states = [state_inner]
    for _ in 2:sampler.num_repeat
        transition, state_inner = AbstractMCMC.step(rng, model, sampler.sampler, state_inner; kwargs...)
        push!(transitions, transition)
        push!(states, state_inner)
    end

    return SequentialTransitions(transitions), SequentialStates(states)
end
