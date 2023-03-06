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
    ordering(sampler)

Return either `ProcessOrdering` or `ChainOrdering` to indicate ordering.
"""
function ordering end

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
# TODO: Should this depend on `orderinge`?
function getparams_and_logprob(state::SwapState)
    # NOTE: Returns parameters, etc. in the chain-ordering, not the process-ordering.
    return getparams_and_logprob(MultipleStates(state.states))
end
function getparams_and_logprob(model, state::SwapState)
    # NOTE: Returns parameters, etc. in the chain-ordering, not the process-ordering.
    return getparams_and_logprob(model, MultipleStates(state.states))
end

function setparams_and_logprob!!(model, state::SwapState, params, logprobs)
    # Use the `MultipleStates`'s implementation to update the underlying states.
    multistate = setparams_and_logprob!!(model, MultipleStates(state.states), params, logprobs)
    # Update the states!
    return @set state.states = multistate.states
end

"""
    process_to_chain(state, I...)

Return the chain index corresponding to the process index `I`.
"""
process_to_chain(state::SwapState, I...) = process_to_chain(state.process_to_chain, I...)
# NOTE: Array impl. is useful for testing.
process_to_chain(proc2chain::AbstractArray, I...) = proc2chain[I...]

"""
    chain_to_process(state, I...)

Return the process index corresponding to the chain index `I`.
"""
chain_to_process(state::SwapState, I...) = chain_to_process(state.chain_to_process, I...)
# NOTE: Array impl. is useful for testing.
chain_to_process(chain2proc::AbstractArray, I...) = chain2proc[I...]

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
state_for_process(states::AbstractArray, I...) = states[I...]

"""
    model_for_chain([ordering, ]sampler, model, state, I...)

Return the model corresponding to the chain indexed by `I...`.
"""
model_for_chain(sampler, model, state, I...) = model_for_chain(ordering(sampler), sampler, model, state, I...)

"""
    model_for_process(sampler, model, state, I...)

Return the model corresponding to the process indexed by `I...`.
"""
model_for_process(sampler, model, state, I...) = model_for_process(ordering(sampler), sampler, model, state, I...)
