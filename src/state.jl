"""
    TemperedState

A general implementation of a state for a [`TemperedSampler`](@ref).

# Fields

$(FIELDS)

# Description

Suppose we're running 3 chains `X`, `Y` and `Z`, that each target a different
tempered density and explore it using an MCMC sampler / kernel, where the (inverse)
temperatures are given by `β = [1.0, 0.05, 0.0025]`.

We are primarily interested in the sequence of states corresponding to `β = 1.0`.

Moreover, suppose we also have 3 workers/processes for which we run these chains in
"parallel" (can also be serial wlog); the other states and chains running in parallel 
will aid in mixing and full exploration of the "cold" target distribution through swap
moves between chains.

We can perform these swaps in two different ways:
1. Swap the the _states_ between each process, i.e. permute `internal`.
2. Swap the _temperatures_ between each process, i.e. permute `chain_order` and
   update `chain_to_inverse_temperature_map`.

(1) is possibly the most intuitive approach since it would allow for the `i`th worker/process
to correspond to the `i`th inverse temperature in our ladder => we could simply return 
the valid samples corresponding to the 1st worker/process/`internal` entry to recover our
target samples. The downside is that we need to move (potentially high-dimensional) states
between the workers/processes to achieve this.

(2) on the other hand does _not_ preserve the direct process-temperature correspondance.
We now need to keep track of which chain is applying which temperature and at each step
provide that temperature to generate a tempered log density for the exploratory kernel to
proceed. This means that we need only transfer a pair of numbers representing the (inverse)
temperatures between workers rather than the full states, as the state will simply be
acted upon by whatever the current relevant (inverse) temperature is for that worker/process.

This implementation follows approach (2).

Here's an exemplar realisation of five steps of sampling and swap-attempts:

```
Chains:    chain_order    chain_to_inverse_temperature_map      is_swap    total_steps
 | | |
 | | |     [1, 2, 3]      {1 => 1.0, 2 => 0.05, 3 => 0.0025}    false      1
  V  |
  Λ  |     [2, 1, 3]      {1 => 0.05, 2 => 1.0, 3 => 0.0025}    true       2
 | | |
 | | |     [2, 1, 3]      {1 => 0.05, 2 => 1.0, 3 => 0.0025}    false      3
 |  V 
 |  Λ      [2, 3, 1]      {1 => 0.0025, 2 => 1.0, 3 => 0.05}    true       4
 | | |
 | | |     [2, 3, 1]      {1 => 0.0025, 2 => 1.0, 3 => 0.05}    false      5
```

"""
@concrete struct TemperedState
    "collection of `(transition, state)` pairs for each chain"
    internal
    "array of length equal to the number of parallel chains / temperatures, chain_order[i] = j tells us the j-th chain is tempered with the `i`th β at this step, i.e. chain_order[1] always returns the valid chain to take a sample from"
    chain_order
    "dict of chain ids and their corresponding β"
    chain_to_inverse_temperature_map
    "total number of steps taken"
    total_steps
    "number of burn-in steps taken"
    burnin_steps
    "contains all necessary information for adaptation of inverse_temperatures"
    adaptation_states
    "array of flags identifying swaps carried out during each step"
    is_swap
    "array of flags identifying which chains are stale, i.e. have not had a succesful proposal since swapping"
    is_stale
    "swap acceptance ratios on log-scale"
    swap_acceptance_ratios
end

get_state(state, I...) = state.internal[I...][2]
get_transition(state, I...) = state.internal[I...][1]
get_transition_params(state, I...) = getparams(get_transition(state, I...))
get_fresh_transition(state) = state.is_stale[state.chain_order[1]] ? nothing : state.internal[state.chain_order[1]][1]
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

"""
    init_state(internal, sampler)

Given an initial `internal` set of transitions and states, instantiate the
`TemperedState` for the run.

- `total_steps` is set to 1 (as the initial steps have already happened)
- `burnin_steps` is set to 0
- `is_swap` and `is_stale` are initially false for all chains

See [`TemperedState`](@ref) for info on the struct that is instantiated.
"""
function init_state(
    internal,
    sampler
)
    return TemperedState(
        internal,
        collect(1:numtemps(sampler)),
        Dict(collect(1:numtemps(sampler)) .=> sampler.inverse_temperatures),
        1,
        0,
        init_adaptation(sampler.adaptation_config, sampler.inverse_temperatures),
        [false for i in 1:numtemps(sampler)],
        [false for i in 1:numtemps(sampler)],
        Dict{Int,Float64}()
    )
end

