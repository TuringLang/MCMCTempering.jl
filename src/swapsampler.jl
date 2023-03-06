"""
    ProcessOrdering

Specifies that the `model` should be treated as process-ordered.
"""
struct ProcessOrdering end

"""
    ChainOrdering

Specifies that the `model` should be treated as chain-ordered.
"""
struct ChainOrdering end

"""
    SwapSampler <: AbstractMCMC.AbstractSampler

# Fields
$(FIELDS)
"""
struct SwapSampler{S,O} <: AbstractMCMC.AbstractSampler
    "swap strategy to use"
    strategy::S
    "ordering assumed for input models"
    model_order::O
end

SwapSampler() = SwapSampler(ReversibleSwap())
SwapSampler(strategy) = SwapSampler(strategy, ChainOrdering())

swapstrategy(sampler::SwapSampler) = sampler.strategy
ordering(::SwapSampler) = ChainOrdering()

"""
    SwapState

A general implementation of a state for a [`TemperedSampler`](@ref).

# Fields

$(FIELDS)

# Description

Suppose we're running 4 chains `X`, `Y`, `Z`, and `W`, each targeting a distribution for different
(inverse) temperatures `β`, say, `1.0`, `0.75`, `0.5`, and `0.25`, respectively. That is, we're mainly 
interested in the chain `(X[1], X[2], … )` which targets the distribution with `β=1.0`.

Moreover, suppose we also have 4 workers/processes for which we run these chains in "parallel"
(can also be serial wlog).

We can then perform a swap in two different ways:
1. Swap the the _states_ between each process, i.e. permute `transitions` and `states`.
2. Swap the _temperatures_ between each process, i.e. permute `chain_to_beta`.

(1) is possibly the most intuitive approach since it means that the i-th worker/process
corresponds to the i-th chain; in this case, process 1 corresponds to `X`, process 2 to `Y`, etc.
The downside is that we need to move (potentially high-dimensional) states between the 
workers/processes.

(2) on the other hand does _not_ preserve the direct process-chain correspondance.
We now need to keep track of which process has which chain, from this we can
reconstruct each of the chains `X`, `Y`, etc. afterwards.
This means that we need only transfer a pair of numbers representing the (inverse)
temperatures between workers rather than the full states.

This implementation follows approach (2).

Here's an example realisation of five steps of sampling and swap-attempts:

```
Chains:    process_to_chain    chain_to_process   inverse_temperatures[process_to_chain[i]]
| | | |       1  2  3  4          1  2  3  4             1.00  0.75  0.50  0.25
| | | |
 V  | |       2  1  3  4          2  1  3  4             0.75  1.00  0.50  0.25
 Λ  | |
| | | |       2  1  3  4          2  1  3  4             0.75  1.00  0.50  0.25
| | | |
|  V  |       2  3  1  4          3  1  2  4             0.75  0.50  1.00  0.25
|  Λ  |
| | | |       2  3  1  4          3  1  2  4             0.75  0.50  1.00  0.25
| | | |
```

In this case, the chain `X` can be reconstructed as:

```julia
X[1] = states[1].states[1]
X[2] = states[2].states[2]
X[3] = states[3].states[2]
X[4] = states[4].states[3]
X[5] = states[5].states[3]
```

and similarly for the states.

The indices here are exactly those represented by `states[k].chain_to_process[1]`.
"""
@concrete struct SwapState
    "collection of states for each process"
    states
    "collection indices such that `chain_to_process[i] = j` if the j-th process corresponds to the i-th chain"
    chain_to_process
    "collection indices such that `process_chain_to[j] = i` if the i-th chain corresponds to the j-th process"
    process_to_chain
    "total number of steps taken"
    total_steps
    "swap acceptance ratios on log-scale"
    swap_acceptance_ratios
end

# TODO: Can we support more?
function SwapState(state::MultipleStates)
    process_to_chain = collect(1:length(state.states))
    chain_to_process = copy(process_to_chain)
    return SwapState(state.states, chain_to_process, process_to_chain, 1, Dict{Int,Float64}())
end

# Defer these to `MultipleStates`.
function getparams_and_logprob(state::SwapState)
    # NOTE: Returns parameters, etc. in the chain-ordering, not the process-ordering.
    return getparams_and_logprob(MultipleStates(map(Base.Fix1(getindex, state.states), state.chain_to_process)))
end
function getparams_and_logprob(model, state::SwapState)
    # NOTE: Returns parameters, etc. in the chain-ordering, not the process-ordering.
    return getparams_and_logprob(model, MultipleStates(map(Base.Fix1(getindex, state.states), state.chain_to_process)))
end

function setparams_and_logprob!!(model, state::SwapState, params, logprobs)
    # Order according to processes.
    process_to_params = map(Base.Fix1(getindex, params), state.process_to_chain)
    process_to_logprobs = map(Base.Fix1(getindex, logprobs), state.process_to_chain)
    # Use the `MultipleStates`'s implementation to update the underlying states.
    multistate = setparams_and_logprob!!(model, MultipleStates(state.states), process_to_params, process_to_logprobs)
    # Update the states!
    return @set state.states = multistate.states
end

process_to_chain(state::SwapState, I...) = process_to_chain(state.process_to_chain, I...)
chain_to_process(state::SwapState, I...) = chain_to_process(state.chain_to_process, I...)
state_for_chain(state::SwapState) = state_for_chain(state, 1)
state_for_chain(state::SwapState, I...) = state_for_process(state, chain_to_process(state, I...))
state_for_process(state::SwapState, I...) = state_for_process(state.states, I...)

function model_for_process(sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    return model_for_process(ordering(sampler), sampler, model, state, I...)
end

function model_for_process(::ProcessOrdering, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to process index, hence we just extract the corresponding index.
    return model.models[I...]
end

function model_for_process(ordering::ChainOrdering, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to chain ordering, hence we need to map the
    # process index `I` to the chain index.
    return model_for_chain(ordering, sampler, model, state, process_to_chain(state, I...))
end

function model_for_chain(sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    return model_for_chain(ordering(sampler), sampler, model, state, I...)
end

function model_for_chain(ordering::ProcessOrdering, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to process index, hence we map chain index to process index
    # and extract the model corresponding to said process.
    return model_for_process(ordering, sampler, model, state, chain_to_process(state, I...))
end

function model_for_chain(::ChainOrdering, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to chain index, hence we just extract the corresponding index.
    return model.models[I...]
end

"""
    SwapTransition

Transition type for tempered samplers.
"""
@concrete struct SwapTransition
    chain_to_process
    process_to_chain
end

# NOTE: This does not have an initial `step`! This is because we need
# states to work with before we can do anything. Hence it only makes
# sense to use this sampler in composition with other samplers.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    sampler::SwapSampler,
    state::SwapState;
    kwargs...
)
    # Reset state
    @set! state.swap_acceptance_ratios = empty(state.swap_acceptance_ratios)

    # Perform a swap step.
    state = swap_step(rng, model, sampler, state)
    @set! state.total_steps += 1

    # We want to return the transition for the _first_ chain, i.e. the chain usually corresponding to `β=1.0`.
    # TODO: What should we return here?
    return SwapTransition(deepcopy(state.chain_to_process), deepcopy(state.process_to_chain)), state
end

# NOTE: The default initial `step` for `CompositionSampler` simply calls the two different
# `step` methods, but since `SwapSampler` does not have such an implementation this will fail.
# Instead we overload the initial `step` for `CompositionSampler` involving `SwapSampler` to
# first take a `step` using the non-swapsampler and then construct `SwapState` from the resulting state.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::CompositionSampler{<:AbstractMCMC.AbstractSampler,<:SwapSampler};
    kwargs...
)
    # This should hopefully be a `MultipleStates` or something since we're working with a `MultiModel`.
    state_outer_initial = last(AbstractMCMC.step(rng, model, outer_sampler(sampler); kwargs...))
    # NOTE: Since `SwapState` wraps a sequence of states from another sampler, we need `state_outer_initial`
    # to initialize the `SwapState`.
    state_inner_initial = SwapState(state_outer_initial)

    # Create the composition state, and take a full step.
    state = composition_state(sampler, state_inner_initial, state_outer_initial)
    return AbstractMCMC.step(rng, model, sampler, state; kwargs...)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::CompositionSampler{<:SwapSampler,<:AbstractMCMC.AbstractSampler};
    kwargs...
)
    # This should hopefully be a `MultipleStates` or something since we're working with a `MultiModel`.
    state_inner_initial = last(AbstractMCMC.step(rng, model, inner_sampler(sampler); kwargs...))
    # NOTE: Since `SwapState` wraps a sequence of states from another sampler, we need `state_outer_initial`
    # to initialize the `SwapState`.
    state_outer_initial = SwapState(state_inner_initial)

    # Create the composition state, and take a full step.
    state = composition_state(sampler, state_inner_initial, state_outer_initial)
    return AbstractMCMC.step(rng, model, sampler, state; kwargs...)
end

@nospecialize function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::CompositionSampler{<:SwapSampler,<:SwapSampler};
    kwargs...
)
    error("`SwapSampler` requires states from sampler other than `SwapSampler` to be initialized")
end

function swap_attempt(rng::Random.AbstractRNG, model::MultiModel, sampler::SwapSampler, state, i, j)
    # Extract the relevant transitions.
    state_i = state_for_chain(state, i)
    state_j = state_for_chain(state, j)
    # Evaluate logdensity for both parameters for each tempered density.
    # NOTE: Assumes ordering of models is according to processes.
    model_i = model_for_chain(sampler, model, state, i)
    model_j = model_for_chain(sampler, model, state, j)
    logπiθi, logπiθj = compute_logdensities(model_i, model_j, state_i, state_j)
    logπjθj, logπjθi = compute_logdensities(model_i, model_j, state_j, state_i)

    # If the proposed temperature swap is accepted according `logα`,
    # swap the temperatures for future steps.
    logα = swap_acceptance_pt(logπiθi, logπiθj, logπjθi, logπjθj)
    should_swap = -Random.randexp(rng) ≤ logα
    if should_swap
        # TODO: Rename `swap_betas!` since no betas are involved anymore?
        swap_betas!(state.chain_to_process, state.process_to_chain, i, j)
    end

    # Keep track of the (log) acceptance ratios.
    state.swap_acceptance_ratios[i] = logα

    # TODO: Handle adaptation.
    return state
end

