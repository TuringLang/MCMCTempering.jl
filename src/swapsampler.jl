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
    "collection of (inverse) temperatures β corresponding to each chain"
    chain_to_beta
    "collection indices such that `chain_to_process[i] = j` if the j-th process corresponds to the i-th chain"
    chain_to_process
    "collection indices such that `process_chain_to[j] = i` if the i-th chain corresponds to the j-th process"
    process_to_chain
    "total number of steps taken"
    total_steps
    "contains all necessary information for adaptation of inverse_temperatures"
    adaptation_states
    "flag which specifies wether this was a swap-step or not"
    is_swap
    "swap acceptance ratios on log-scale"
    swap_acceptance_ratios
end

"""
    process_to_chain(state, I...)

Return the chain index corresponding to the process index `I`.
"""
process_to_chain(state::SwapState, I...) = process_to_chain(state.process_to_chain, I...)
# NOTE: Array impl. is useful for testing.
# process_to_chain(proc2chain::AbstractArray, I...) = proc2chain[I...]

"""
    chain_to_process(state, I...)

Return the process index corresponding to the chain index `I`.
"""
chain_to_process(state::SwapState, I...) = chain_to_process(state.chain_to_process, I...)
# NOTE: Array impl. is useful for testing.
# chain_to_process(chain2proc::AbstractArray, I...) = chain2proc[I...]

"""
    transition_for_chain(state, transitions[, I...])

Return the transition corresponding to the chain indexed by `I...`.
If `I...` is not specified, the transition corresponding to `β=1.0` will be returned, i.e. `I = (1, )`.
"""
transition_for_chain(state::SwapState, transitions) = transition_for_chain(state, transitions, 1)
function transition_for_chain(state::SwapState, transitions, I...)
    return transition_for_process(state, transitions, chain_to_process(state, I...))
end

"""
    transition_for_process(state, transitions, I...)

Return the transition corresponding to the process indexed by `I...`.
"""
transition_for_process(state::SwapState, transitions, I...) = transition_for_process(transitions, I...)
# transition_for_process(transitions, I...) = transitions[I...]

"""
    state_for_chain(state[, I...])

Return the state corresponding to the chain indexed by `I...`.
If `I...` is not specified, the state corresponding to `β=1.0` will be returned.
"""
state_for_chain(state::SwapState) = state_for_chain(state, 1)
state_for_chain(state::SwapState, I...) = state_for_process(state, chain_to_process(state, I...))

"""
    state_for_process(state, I...)

Return the state corresponding to the process indexed by `I...`.
"""
state_for_process(state::SwapState, I...) = state_for_process(state.states, I...)
# state_for_process(states, I...) = states[I...]

"""
    beta_for_chain(state[, I...])

Return the β corresponding to the chain indexed by `I...`.
If `I...` is not specified, the β corresponding to `β=1.0` will be returned.
"""
beta_for_chain(state::SwapState) = beta_for_chain(state, 1)
beta_for_chain(state::SwapState, I...) = beta_for_chain(state.chain_to_beta, I...)
# NOTE: Array impl. is useful for testing.
# beta_for_chain(chain_to_beta::AbstractArray, I...) = chain_to_beta[I...]

"""
    beta_for_process(state, I...)

Return the β corresponding to the process indexed by `I...`.
"""
beta_for_process(state::SwapState, I...) = beta_for_process(state.chain_to_beta, state.process_to_chain, I...)
# NOTE: Array impl. is useful for testing.
# function beta_for_process(chain_to_beta::AbstractArray, proc2chain::AbstractArray, I...)
#     return beta_for_chain(chain_to_beta, process_to_chain(proc2chain, I...))
# end

# """
#     model_for_chain(sampler, model, state, I...)

# Return the model corresponding to the chain indexed by `I...`.
# """
# function model_for_chain(sampler, model, state, I...)
#     return make_tempered_model(sampler, model, beta_for_chain(state, I...))
# end

# """
#     model_for_process(sampler, model, state, I...)

# Return the model corresponding to the process indexed by `I...`.
# """
# function model_for_process(sampler, model, state, I...)
#     return make_tempered_model(sampler, model, beta_for_process(state, I...))
# end

# HACK: Remove this.
state_from(model, swapstate::SwapState, state) = error("no")
function state_from(model, swapstate::SwapState, multistate::MultipleStates)
    @assert length(swapstate.states) == length(multistate.states) "number of states ($(length(swapstate.states)) and $(length(multistate.states))) does not match"
    states = map(swapstate.states, multistate.states) do state_from_swap, state_from_multi
        state_from(model, state_from_swap, state_from_multi)
    end
    return @set swapstate.states = states
end

"""
    SwapTransition

Transition type for tempered samplers.
"""
struct SwapTransition{S}
    transition::S
end

getparams_and_logprob(transition::SwapTransition) = getparams_and_logprob(transition.transition)
getparams_and_logprob(model, transition::SwapTransition) = getparams_and_logprob(model, transition.transition)


# AbstractMCMC interface
using AbstractMCMC: AbstractMCMC

struct SwapSampler{S} <: AbstractMCMC.AbstractSampler
    strategy::S
end

SwapSampler() = SwapSampler(ReversibleSwap())

swapstrategy(sampler::SwapSampler) = sampler.strategy

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
    @set! state.is_swap = true
    @set! state.total_steps += 1

    # We want to return the transition for the _first_ chain, i.e. the chain usually corresponding to `β=1.0`.
    # TODO: What should we return here?
    return SwapTransition(chain_to_process(state)), state
end

# Tempered sampler.
@concrete struct TemperedComposition <: AbstractMCMC.AbstractSampler
    "sampler to use for swapping"
    swapsampler
    "sampler(s) used to target the tempered distributions"
    sampler
    "collection of inverse temperatures β; β[i] correponds i-th tempered model"
    inverse_temperatures
    "the swap strategy that will be used when proposing swaps"
    swap_strategy
    # TODO: This should be replaced with `P` just being some `NoAdapt` type.
    "boolean flag specifying whether or not to adapt"
    adapt
    "adaptation parameters"
    adaptation_states
end

function TemperedComposition(swapsampler, sampler, inverse_temperatures)
    return TemperedComposition(swapsampler, sampler, inverse_temperatures, ReversibleSwap(), false, nothing)
end

numtemps(sampler::TemperedComposition) = length(sampler.inverse_temperatures)

# TODO: Improve.
getsampler(sampler::TemperedComposition, I...) = getsampler(sampler.sampler, I...)

# TODO: Make this configurable.
saveall(sampler::TemperedComposition) = true

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedComposition;
    kwargs...
)
    # Create a `MultiSampler` and `MultiModel`.
    multimodel = MultiModel([
        make_tempered_model(sampler, model, sampler.inverse_temperatures[i])
        for i in 1:numtemps(sampler)
    ])
    multisampler = MultiSampler([getsampler(sampler, i) for i in 1:numtemps(sampler)])
    @info "heyo 1" multimodel multisampler
    multistate = last(AbstractMCMC.step(rng, multimodel, multisampler; kwargs...))
    @info "heyo 2"

    # Make sure to collect, because we'll be using `setindex!(!)` later.
    process_to_chain = collect(1:length(sampler.inverse_temperatures))
    # Need to `copy` because this might be mutated.
    chain_to_process = copy(process_to_chain)
    swapstate = SwapState(
        multistate.states,
        sampler.inverse_temperatures,
        chain_to_process,
        process_to_chain,
        1,
        sampler.adaptation_states,
        false,
        Dict{Int,Float64}()
    )

    @info "heyo 3"
    return AbstractMCMC.step(rng, model, sampler, composition_state(sampler, swapstate, multistate))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedComposition,
    state;
    kwargs...
)
    @info "heyo 4"
    # Get the samplers.
    swapsampler = sampler.swapsampler
    # Extract the previous states.
    swapstate_prev, multistate_prev = inner_state(state), outer_state(state)

    # TODO: `SwapSampler` should probably only act on `MultiModel`.
    multimodel_swap = MultiModel([model_for_process(sampler, model, state, i) for i in 1:numtemps(sampler)])
    multisampler_swap = MultiSampler([swapstrategy(sampler) for i in 1:numtemps(sampler)])

    # Update the `swapstate`.
    swapstate = state_from(model, swapstate_prev, multistate_prev)
    @info "heyo 5"
    # Take a step with the swap sampler.
    swaptransition, swapstate = AbstractMCMC.step(rng, multimodel_swap, swapsampler, swapstate; kwargs...)
    @info "heyo 6"
    # Create the multi-versions with the ordering corresponding to the processes.
    multimodel = MultiModel([model_for_process(sampler, model, state, i) for i in 1:numtemps(sampler)])
    multisampler = MultiSampler([sampler_for_process(sampler, state, i) for i in 1:numtemps(sampler)])
    multistate = MultipleStates([state_for_process(state, i) for i in 1:numtemps(sampler)])

    # Take a step with the multi sampler.
    multitransition, multistate = AbstractMCMC.step(rng, multimodel, multisampler, multistate; kwargs...)

    return (
        composition_transition(sampler, swaptransition, multitransition),
        composition_state(sampler, swapstate, multistate)
    )
end
