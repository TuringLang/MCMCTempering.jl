"""
    TemperedState

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
1. Swap the the _states_ between each process, i.e. permute `transitions_and_states`.
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
X[1] = states[1].transitions_and_states[1]
X[2] = states[2].transitions_and_states[2]
X[3] = states[3].transitions_and_states[2]
X[4] = states[4].transitions_and_states[3]
X[5] = states[5].transitions_and_states[3]
```

The indices here are exactly those represented by `states[k].chain_to_process[1]`.
"""
@concrete struct TemperedState
    "collection of `(transition, state)` pairs for each process"
    transitions_and_states
    "collection of (inverse) temperatures β corresponding to each chain"
    chain_to_beta
    "collection indices such that `chain_to_process[i] = j` if the j-th process corresponds to the i-th chain"
    chain_to_process
    "collection indices such that `process_chain_to[j] = i` if the i-th chain corresponds to the j-th process"
    process_to_chain
    "total number of steps taken"
    total_steps
    "number of burn-in steps taken"
    burnin_steps
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
process_to_chain(state::TemperedState, I...) = process_to_chain(state.process_to_chain, I...)
# NOTE: Array impl. is useful for testing.
process_to_chain(proc2chain::AbstractArray, I...) = proc2chain[I...]

"""
    chain_to_process(state, I...)

Return the process index corresponding to the chain index `I`.
"""
chain_to_process(state::TemperedState, I...) = chain_to_process(state.chain_to_process, I...)
# NOTE: Array impl. is useful for testing.
chain_to_process(chain2proc::AbstractArray, I...) = chain2proc[I...]

"""
    transition_for_chain(state[, I...])

Return the transition corresponding to the chain indexed by `I...`.
If `I...` is not specified, the transition corresponding to `β=1.0` will be returned, i.e. `I = (1, )`.
"""
transition_for_chain(state::TemperedState) = transition_for_chain(state, 1)
transition_for_chain(state::TemperedState, I...) = transition_for_process(state, chain_to_process(state, I...))

"""
    transition_for_process(state, I...)

Return the transition corresponding to the process indexed by `I...`.
"""
transition_for_process(state::TemperedState, I...) = state.transitions_and_states[I...][1]

"""
    state_for_chain(state[, I...])

Return the state corresponding to the chain indexed by `I...`.
If `I...` is not specified, the state corresponding to `β=1.0` will be returned.
"""
state_for_chain(state::TemperedState) = state_for_chain(state, 1)
state_for_chain(state::TemperedState, I...) = state_for_process(state, chain_to_process(state, I...))

"""
    state_for_process(state, I...)

Return the state corresponding to the process indexed by `I...`.
"""
state_for_process(state::TemperedState, I...) = state.transitions_and_states[I...][2]

"""
    beta_for_chain(state[, I...])

Return the β corresponding to the chain indexed by `I...`.
If `I...` is not specified, the β corresponding to `β=1.0` will be returned.
"""
beta_for_chain(state::TemperedState) = beta_for_chain(state, 1)
beta_for_chain(state::TemperedState, I...) = beta_for_chain(state.chain_to_beta, I...)
# NOTE: Array impl. is useful for testing.
beta_for_chain(chain_to_beta::AbstractArray, I...) = chain_to_beta[I...]

"""
    beta_for_process(state, I...)

Return the β corresponding to the process indexed by `I...`.
"""
beta_for_process(state::TemperedState, I...) = beta_for_process(state.chain_to_beta, state.process_to_chain, I...)
# NOTE: Array impl. is useful for testing.
function beta_for_process(chain_to_beta::AbstractArray, proc2chain::AbstractArray, I...)
    return beta_for_chain(chain_to_beta, process_to_chain(proc2chain, I...))
end

"""
    getparams(transition)
    getparams(::Type, transition)

Return the parameters contained in `transition`.

If a type is specified, the parameters are returned in said type.

# Notes
This method is meant to be overloaded for the different transitions types.
"""
function getparams end
