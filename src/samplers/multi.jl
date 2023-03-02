# Multiple independent samplers.
combine(x::Tuple, y::Tuple) = (x..., y...)
combine(x::Tuple, y) = (x..., y)
combine(x, y::Tuple) = (x, y...)
combine(x::AbstractArray, y::AbstractArray) = vcat(x, y)
combine(x::AbstractArray, y) = vcat(x, y)
combine(x, y::AbstractArray) = vcat(x, y)
combine(x, y) = Iterators.flatten((x, y))


"""
    MultiSampler <: AbstractMCMC.AbstractSampler

A `MultiSampler` is a container for multiple samplers.

See also: [`MultiModel`](@ref).

# Fields
$(FIELDS)

# Examples
```julia
# `sampler1` targets `model1`, `sampler2` targets `model2`, etc.
multi_model = model1 × model2 × model3 # or `MultiModel((model1, model2, model3))`
multi_sampler = sampler1 × sampler2 × sampler3 # or `MultiSampler((sampler1, sampler2, sampler3))`
# Target the joint model.
AbstractMCMC.step(rng, multi_model, multi_sampler)
```
"""
struct MultiSampler{S} <: AbstractMCMC.AbstractSampler
    "The samplers"
    samplers::S
end

×(sampler1::AbstractMCMC.AbstractSampler, sampler2::AbstractMCMC.AbstractSampler) = MultiSampler((sampler1, sampler2))
×(sampler1::MultiSampler, sampler2::AbstractMCMC.AbstractSampler) = MultiSampler(combine(sampler1.samplers, sampler2))
×(sampler1::AbstractMCMC.AbstractSampler, sampler2::MultiSampler) = MultiSampler(combine(sampler1, sampler2.samplers))
×(sampler1::MultiSampler, sampler2::MultiSampler) = MultiSampler(combine(sampler1.samplers, sampler2.samplers))

"""
    MultiModel <: AbstractMCMC.AbstractModel

A `MultiModel` is a container for multiple models.

See also: [`MultiSampler`](@ref).

# Fields
$(FIELDS)
"""
struct MultiModel{M} <: AbstractMCMC.AbstractModel
    "The models"
    models::M
end

×(model1::AbstractMCMC.AbstractModel, model2::AbstractMCMC.AbstractModel) = MultiModel((model1, model2))
×(model1::MultiModel, model2::AbstractMCMC.AbstractModel) = MultiModel(combine(model1.models, model2))
×(model1::AbstractMCMC.AbstractModel, model2::MultiModel) = MultiModel(combine(model1, model2.models))
×(model1::MultiModel, model2::MultiModel) = MultiModel(combine(model1.models, model2.models))

# TODO: Make these subtypes of `AbstractVector`?
"""
    MultipleTransitions

A container for multiple transitions.

See also: [`MultipleStates`](@ref).

# Fields
$(FIELDS)
"""
struct MultipleTransitions{A}
    "The transitions"
    transitions::A
end

"""
    MultipleStates

A container for multiple states.

See also: [`MultipleTransitions`](@ref).

# Fields
$(FIELDS)
"""
struct MultipleStates{A}
    "The states"
    states::A
end

# NOTE: This is different from most of the other implementations of `getparams_and_logprob`
# as here we need to work with multiple models, transitions, and states.
function getparams_and_logprob(model::MultiModel, state::MultipleStates)
    params_and_logprobs = map(getparams_and_logprob, model.models, state.states)
    return map(first, params_and_logprobs), map(last, params_and_logprobs)
end
function setparams_and_logprob!!(model::MultiModel, state::MultipleStates, params, logprob)
    @assert length(params) == length(logprob) == length(state.states) "The number of parameters and log probabilities must match the number of states."
    return @set state.states = map(setparams_and_logprob!!, model.models, state.states, params, logprob)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::MultiSampler;
    kwargs...
)
    @assert length(model.models) == length(sampler.samplers) "Number of models and samplers must be equal"
    transition_and_states = asyncmap(model.models, sampler.samplers) do model, sampler
        AbstractMCMC.step(rng, model, sampler; kwargs...)
    end

    return MultipleTransitions(map(first, transition_and_states)), MultipleStates(map(last, transition_and_states))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::MultiSampler,
    states::MultipleStates;
    kwargs...
)
    @assert length(model.models) == length(sampler.samplers) == length(states.states) "Number of models, samplers, and states must be equal."

    transition_and_states = asyncmap(model.models, sampler.samplers, states.states) do model, sampler, state
        AbstractMCMC.step(rng, model, sampler, state; kwargs...)
    end

    return MultipleTransitions(map(first, transition_and_states)), MultipleStates(map(last, transition_and_states))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::RepeatedSampler{<:MultiSampler},
    states::MultipleStates;
    kwargs...
)
    multisampler = sampler.sampler
    @assert length(model.models) == length(multisampler.samplers) == length(states.states) "Number of models, samplers, and states must be equal."
    transition_and_states = asyncmap(model.models, multisampler.samplers, states.states) do model, sampler, state
        # Just re-wrap each of the samplers in a `RepeatedSampler` and call it's implementation.
        AbstractMCMC.step(rng, model, RepeatedSampler(sampler, sampler.num_repeat), state; kwargs...)
    end

    return MultipleTransitions(map(first, transition_and_states)), MultipleStates(map(last, transition_and_states))
end
