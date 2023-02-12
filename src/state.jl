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
1. Swap the the _states_ between each process, i.e. permute `internal_state`.
2. Swap the _temperatures_ between each process, i.e. permute `inverse_temperatures`.

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

Here's an exemplar realisation of five steps of sampling and swap-attempts:

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
X[1] = states[1].internal_state[1]
X[2] = states[2].internal_state[2]
X[3] = states[3].internal_state[2]
X[4] = states[4].internal_state[3]
X[5] = states[5].internal_state[3]
```

The indices here are exactly those represented by `states[k].chain_to_process[1]`.
"""
@concrete struct TemperedState
    "collection of `(transition, state)` pairs for each chain"
    internal_state
    "array of length equal to the number of parallel chains / temperatures, chain_order[i] = j tells us the j-th chain is tempered with the i-th β at this step, i.e. chain_order[1] always returns the valid chain to take a sample from"
    chain_order
    "dict of chain ids and their corresponding β"
    chain_to_inverse_temperature_map
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

get_state(state) = state.internal_state[state.chain_order[1]][2]
get_state(state, I...) = state.internal_state[I...][2]
get_transition(state) = state.internal_state[state.chain_order[1]][1]
get_transition(state, I...) = state.internal_state[I...][1]
get_transition_params(state, I...) = getparams(get_transition(state, I...))
get_inverse_temperature(state, I...) = state.chain_to_inverse_temperature_map[I...]

"""
    getparams(transition)
    getparams(::Type, transition)

Return the parameters contained in `transition`.

If a type is specified, the parameters are returned in said type.

# Notes
This method is meant to be overloaded for the different transitions types.
"""
function getparams end

function init_state(internal_state, chain_order, sampler)
    return TemperedState(
        internal_state,
        chain_order,
        Dict(chain_order .=> sampler.inverse_temperatures),
        1,
        0,
        sampler.adaptation_config.schedule == NoAdapt() ? nothing : init_adaptation(sampler.adaptation_config, sampler.inverse_temperatures),
        false,
        Dict{Int,Float64}()
    )
end

