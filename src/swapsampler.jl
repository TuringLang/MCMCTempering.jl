"""
    ProcessOrder

Specifies that the `model` should be treated as process-ordered.
"""
struct ProcessOrder end

"""
    ChainOrder

Specifies that the `model` should be treated as chain-ordered.
"""
struct ChainOrder end

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
# TODO: What is the best way to implement these? Should we sort according to the chain indices
# to match the order of the models?
# getparams_and_logprob(state::SwapState) = getparams_and_logprob(MultipleStates(state.states))
# getparams_and_logprob(model, state::SwapState) = getparams_and_logprob(model, MultipleStates(state.states))

function setparams_and_logprob!!(model, state::SwapState, params, logprobs)
    # Use the `MultipleStates`'s implementation to update the underlying states.
    multistate = setparams_and_logprob!!(model, MultipleStates(state.states), params, logprobs)
    # Update the states!
    return @set state.states = multistate.states
end

"""
    sort_by_chain(::ChainOrdering, state, xs)
    sort_by_chain(::ProcessOrdering, state, xs)

Return `xs` sorted according to the chain indices, as specified by `state`.
"""
sort_by_chain(::ChainOrder, ::Any, xs) = xs
sort_by_chain(::ProcessOrder, state, xs) = [xs[chain_to_process(state, i)] for i = 1:length(xs)]
sort_by_chain(::ProcessOrder, state, xs::Tuple) = ntuple(i -> xs[chain_to_process(state, i)], length(xs))

"""
    sort_by_process(::ProcessOrdering, state, xs)
    sort_by_process(::ChainOrdering, state, xs)

Return `xs` sorted according to the process indices, as specified by `state`.
"""
sort_by_process(::ProcessOrder, ::Any, xs) = xs
sort_by_process(::ChainOrder, state, xs) = [xs[process_to_chain(state, i)] for i = 1:length(xs)]
sort_by_process(::ChainOrder, state, xs::Tuple) = ntuple(i -> xs[process_to_chain(state, i)], length(xs))

"""
    process_to_chain(state, I...)

Return the chain index corresponding to the process index `I`.
"""
process_to_chain(state::SwapState, I...) = process_to_chain(state.process_to_chain, I...)
# NOTE: Array impl. is useful for testing.
process_to_chain(proc2chain, I...) = proc2chain[I...]

"""
    chain_to_process(state, I...)

Return the process index corresponding to the chain index `I`.
"""
chain_to_process(state::SwapState, I...) = chain_to_process(state.chain_to_process, I...)
# NOTE: Array impl. is useful for testing.
chain_to_process(chain2proc, I...) = chain2proc[I...]

"""
    state_for_chain(state[, I...])

Return the state corresponding to the chain indexed by `I...`.
If `I...` is not specified, the state corresponding to `β=1.0` will be returned.
"""
state_for_chain(state) = state_for_chain(state, 1)
state_for_chain(state, I...) = state_for_process(state, chain_to_process(state, I...))

"""
    state_for_process(state, I...)

Return the state corresponding to the process indexed by `I...`.
"""
state_for_process(state::SwapState, I...) = state_for_process(state.states, I...)
state_for_process(proc2state, I...) = proc2state[I...]

"""
    model_for_chain(ordering, sampler, model, state, I...)

Return the model corresponding to the chain indexed by `I...`.

`ordering` specifies what sort of order the input models follow.
"""
function model_for_chain end

"""
    model_for_process(ordering, sampler, model, state, I...)

Return the model corresponding to the process indexed by `I...`.

`ordering` specifies what sort of order the input models follow.
"""
function model_for_process end

"""
    models_by_processes(ordering, models, state)

Return the models in the order of processes, assuming `models` is sorted according to `ordering`.

See also: [`ProcessOrdering`](@ref), [`ChainOrdering`](@ref).
"""
models_by_processes(ordering, models, state) = sort_by_process(ordering, state, models)

"""
    samplers_by_processes(ordering, samplers, state)

Return the `samplers` in the order of processes, assuming `samplers` is sorted according to `ordering`.

See also: [`ProcessOrdering`](@ref), [`ChainOrdering`](@ref).
"""
samplers_by_processes(ordering, samplers, state) = sort_by_process(ordering, state, samplers)

"""
    SwapSampler <: AbstractMCMC.AbstractSampler

# Fields
$(FIELDS)
"""
struct SwapSampler{S} <: AbstractMCMC.AbstractSampler
    "swap strategy to use"
    strategy::S
end

SwapSampler() = SwapSampler(ReversibleSwap())

"""
    swapstrategy(sampler::SwapSampler)

Return the swap-strategy used by `sampler`.
"""
swapstrategy(sampler::SwapSampler) = sampler.strategy

# Interaction with the state.
# NOTE: `SwapSampler` should only every interact with `ProcessOrdering`, so we don't implement `ChainOrdering`.
function model_for_chain(ordering::ProcessOrder, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to process index, hence we map chain index to process index
    # and extract the model corresponding to said process.
    return model_for_process(ordering, sampler, model, state, chain_to_process(state, I...))
end

function model_for_process(::ProcessOrder, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to process index, hence we just extract the corresponding index.
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

chain_to_process(state::SwapTransition, I...) = chain_to_process(state.chain_to_process, I...)
process_to_chain(state::SwapTransition, I...) = process_to_chain(state.process_to_chain, I...)

function composition_transition(
    sampler::CompositionSampler{<:AbstractMCMC.AbstractSampler,<:SwapSampler},
    swaptransition::SwapTransition,
    outertransition::MultipleTransitions
)
    saveall(sampler) && return CompositionTransition(outertransition, swaptransition)
    # Otherwise we have to re-order the transitions, since without the `swaptransition` there's
    # no way to recover the true ordering of the transitions.
    return MultipleTransitions(sort_by_chain(ProcessOrder(), swaptransition, outertransition.transitions))
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

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::CompositionSampler{<:AbstractMCMC.AbstractSampler,<:SwapSampler},
    state;
    kwargs...
)
    # Reminder: a `swap` can be implemented in two different ways:
    #
    # 1. Swap the models and leave ordering of (sampler, state)-pair unchanged.
    # 2. Swap (sampler, state)-pairs and leave ordering of models unchanged.
    #
    # (1) has the properties:
    # + Easy to keep `outerstate` and `swapstate` in sync since their ordering is never changed.
    # - Ordering of `outerstate` no longer corresponds to ordering of models, i.e. the returned
    #   `outerstate.states[i]` does no longer correspond to a state targeting `model.models[i]`.
    #   This will have to be adjusted in the `AbstractMCMC.bundle_samples` before, say, converting
    #   into a `MCMCChains.Chains`.
    #
    # (2) has the properties:
    # + Returned `outertransition` (and `outerstate`, if we want) has the same ordering as the models,
    #   i.e. `outerstate.states[i]` now corresponds to `model.models[i]`!
    # - Need to keep `outerstate` and `swapstate` in sync since their ordering now changes.
    # - Need to also re-order `outersampler.samplers` :/
    #
    # Here (as in, below) we go with option (1), i.e. re-order the `models`.
    # A full `step` then is as follows:
    # 1. Sort models according to index processes using the `swapstate` from previous iteration.
    # 2. Take step with `swapsampler`.
    # 3. Sort models _again_ according to index processes using the new `swapstate`, since we
    #    might have made a swap in (2).
    # 4. Run multi-sampler.

    outersampler, swapsampler = outer_sampler(sampler), inner_sampler(sampler)

    # Get the states.
    outerstate_prev, swapstate_prev = outer_state(state), inner_state(state)

    # Re-order the models.
    chain2models = model.models  # but keep the original chain → model around because we'll re-order again later
    @set! model.models = models_by_processes(ChainOrder(), chain2models, swapstate_prev)

    # Step for the swap-sampler.
    # TODO: We should probably call `state_from(model, model_other, state, state_other)` so we
    # can avoid additional log-joint computations, gradient commputations, etc.
    swaptransition, swapstate = AbstractMCMC.step(
        rng, model, swapsampler, state_from(model, model, swapstate_prev, outerstate_prev);
        kwargs...
    )

    # Re-order the models AGAIN, since we might have swapped some.
    # NOTE: We don't override `model` because we want to let `state_from` know that the previous
    # states came from `model`, not `model_reordered`.
    model_reordered = @set model.models = models_by_processes(ChainOrder(), chain2models, swapstate)

    # Create the current state from `outerstate_prev` and `swapstate`, and `step` for `outersampler`.`
    outertransition, outerstate = AbstractMCMC.step(
        # HACK: We really need the `state_from` here despite the fact that `SwapSampler` does note
        # change the `swapstates.states` itself, but we might require a re-computation of certain
        # quantities from the `model`, which has now potentially been re-ordered (see above).
        # NOTE: We do NOT do `state_from(model, outerstate_prev, swapstate)` because as of now,
        # `swapstate` does not implement `getparams_and_logprob`.
        rng, model_reordered, outersampler, state_from(model_reordered, model, outerstate_prev, outerstate_prev);
        kwargs...
    )

    # TODO: Should we re-order the transitions?
    # Currently, one has to re-order the `outertransition` according to `swaptransition`
    # in the `bundle_samples`. Is this the right approach though?
    # TODO: We should at least re-order transitions in the case where `saveall(sampler) == false`!
    # In this case, we'll just return the transition without the swap-transition, hence making it
    # impossible to reconstruct the actual ordering!
    return (
        composition_transition(sampler, swaptransition, outertransition),
        composition_state(sampler, swapstate, outerstate)
    )
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

"""
    swap_attempt(rng, model, sampler, state, i, j)

Attempt to swap the temperatures of two chains by tempering the densities and
calculating the swap acceptance ratio; then swapping if it is accepted.
"""
function swap_attempt(rng::Random.AbstractRNG, model::MultiModel, sampler::SwapSampler, state, i, j)
    # Extract the relevant transitions.
    state_i = state_for_chain(state, i)
    state_j = state_for_chain(state, j)
    # Evaluate logdensity for both parameters for each tempered density.
    # NOTE: `SwapSampler` should only be working with models ordered according to `ProcessOrder`,
    # never `ChainOrder`, hence why we have the below.
    model_i = model_for_chain(ProcessOrder(), sampler, model, state, i)
    model_j = model_for_chain(ProcessOrder(), sampler, model, state, j)
    logπiθi, logπiθj = compute_logdensities(model_i, model_j, state_i, state_j)
    logπjθj, logπjθi = compute_logdensities(model_j, model_i, state_j, state_i)

    # If the proposed temperature swap is accepted according `logα`,
    # swap the temperatures for future steps.
    logα = swap_acceptance_pt(logπiθi, logπiθj, logπjθi, logπjθj)
    should_swap = -Random.randexp(rng) ≤ logα
    if should_swap
        swap!(state.chain_to_process, state.process_to_chain, i, j)
    end

    # Keep track of the (log) acceptance ratios.
    state.swap_acceptance_ratios[i] = logα

    # TODO: Handle adaptation.
    return state
end

"""
    swap_step([strategy::AbstractSwapStrategy, ]rng, model, sampler, state)

Return a new `state`, with temperatures possibly swapped according to `strategy`.

If no `strategy` is provided, the return-value of [`swapstrategy`](@ref) called on `sampler`
is used.
"""
function swap_step(
    rng::Random.AbstractRNG,
    model,
    sampler,
    state
)
    return swap_step(swapstrategy(sampler), rng, model, sampler, state)
end

function swap_step(
    strategy::ReversibleSwap,
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler,
    state
)
    # Randomly select whether to attempt swaps between chains
    # corresponding to odd or even indices of the temperature ladder
    odd = rand(rng, Bool)
    # TODO: Use integer-division.
    for k in [Int(2 * i - odd) for i in 1:(floor((length(model) - 1 + odd) / 2))]
        state = swap_attempt(rng, model, sampler, state, k, k + 1)
    end
    return state
end


function swap_step(
    strategy::NonReversibleSwap,
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler,
    state::SwapState  # we're accessing `total_steps` restrict the type here
)
    # Alternate between attempting to swap chains corresponding
    # to odd and even indices of the temperature ladder
    odd = state.total_steps % 2 != 0
    # TODO: Use integer-division.
    for k in [Int(2 * i - odd) for i in 1:(floor((length(model) - 1 + odd) / 2))]
        state = swap_attempt(rng, model, sampler, state, k, k + 1)
    end
    return state
end

function swap_step(
    strategy::SingleSwap,
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler,
    state
)
    # Randomly pick one index `k` of the temperature ladder and
    # attempt a swap between the corresponding chain and its neighbour
    k = rand(rng, 1:(length(model) - 1))
    return swap_attempt(rng, model, sampler, state, k, k + 1)
end

function swap_step(
    strategy::SingleRandomSwap,
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler,
    state
)
    # Randomly pick two temperature ladder indices in order to
    # attempt a swap between the corresponding chains
    chains = Set(1:length(model))
    i = pop!(chains, rand(rng, chains))
    j = pop!(chains, rand(rng, chains))
    return swap_attempt(rng, model, sampler, state, i, j)
end

function swap_step(
    strategy::RandomSwap,
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler,
    state
)
    # Iterate through all of temperature ladder indices, picking random
    # pairs and attempting swaps between the corresponding chains
    chains = Set(1:length(model))
    while length(chains) >= 2
        i = pop!(chains, rand(rng, chains))
        j = pop!(chains, rand(rng, chains))
        state = swap_attempt(rng, model, sampler, state, i, j)
    end
    return state
end

function swap_step(
    strategy::NoSwap,
    rng::Random.AbstractRNG,
    model,
    sampler,
    state
)
    return state
end


