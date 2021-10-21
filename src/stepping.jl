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
(it can also be in serial, but for the sake of demonstration imagine it's parallel).

When can then perform a swap in two different ways:
1. Swap the the _states_ between each process, i.e. permute `transitions_and_states`.
2. Swap the _temperatures_ between each process, i.e. permute `inverse_temperatures`.

(1) is possibly the most intuitive approach since it means that the i-th worker/process
corresponds to the i-th chain; in this case, process 1 corresponds to `X`, process 2 to `Y`, etc.
The downside is that we need to move (potentially high-dimensional) states between the 
workers/processes.

(2) on the other hand does _not_ preserve the direct process-chain correspondance.
We now need to keep track of which process has which chain, from this we can
reconstruct each of the chains `X`, `Y`, etc. afterwards.
On the other hand, this means that we only need to transfer a pair of numbers 
representing the (inverse) temperatures between workers rather than the full states.

The current implementation follows approach (2).

Here's an example realization of using five steps of sampling and swap-attempts:

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
    "collection of (inverse) temperatures β corresponding to each process"
    inverse_temperatures
    "collection indices such that `chain_to_process[i] = j` if the j-th process corresponds to the i-th chain"
    chain_to_process
    "collection indices such that `process_chain_to[j] = i` if the i-th chain corresponds to the j-th process"
    process_to_chain
    "total number of steps taken"
    total_steps
    "contains all necessary information for adaptation of inverse_temperatures"
    Ρ
end

"""
    transition_for_chain(state[, I...])

Return the transition corresponding to the chain indexed by `I...`.
If `I...` is not specified, the transition corresponding to `β=1.0` will be returned, i.e. `I = (1, )`.
"""
transition_for_chain(state::TemperedState) = transition_for_chain(state, 1)
transition_for_chain(state::TemperedState, I...) = state.transitions_and_states[state.chain_to_process[I...]][1]

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
state_for_chain(state::TemperedState, I...) = state.transitions_and_states[I...][2]

"""
    state_for_process(state, I...)

Return the state corresponding to the process indexed by `I...`.
"""
state_for_process(state::TemperedState, I...) = state.transitions_and_states[I...][2]

"""
    β_for_chain(state[, I...])

Return the β corresponding to the chain indexed by `I...`.
If `I...` is not specified, the β corresponding to `β=1.0` will be returned.
"""
β_for_chain(state::TemperedState) = β_for_chain(state, 1)
β_for_chain(state::TemperedState, I...) = state.inverse_temperatures[state.chain_to_process[I...]]

"""
    β_for_process(state, I...)

Return the β corresponding to the process indexed by `I...`.
"""
β_for_process(state::TemperedState, I...) = state.inverse_temperatures[I...]

"""
    sampler_for_chain(sampler::TemperedSampler, state::TemperedState[, I...])

Return the sampler corresponding to the chain indexed by `I...`.
If `I...` is not specified, the sampler corresponding to `β=1.0` will be returned.
"""
sampler_for_chain(sampler::TemperedSampler, state::TemperedState) = sampler_for_chain(sampler, state, 1)
function sampler_for_chain(sampler::TemperedSampler, state::TemperedState, I...)
    return getsampler(sampler.sampler, state.chain_to_process[I...])
end

"""
    sampler_for_process(sampler::TemperedSampler, state::TemperedState, I...)

Return the sampler corresponding to the process indexed by `I...`.
"""
function sampler_for_process(sampler::TemperedSampler, state::TemperedState, I...)
    return getsampler(sampler.sampler, I...)
end

"""
    should_swap(sampler, state)

Return `true` if a swap should happen at this iteration, and `false` otherwise.
"""
function should_swap(sampler::TemperedSampler, state::TemperedState)
    return state.total_steps % sampler.swap_every == 0
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler;
    init_params=nothing,
    kwargs...
)
    # `TemperedState` has the transitions and states in the order of
    # the processes, and performs swaps by moving the (inverse) temperatures
    # `β` between the processes, rather than moving states between processes
    # and keeping the `β` local to each process.
    # 
    # Therefore we iterate over the processes and then extract the corresponding
    # `β`, `sampler` and `state`, and take a initialize.
    transitions_and_states = [
        AbstractMCMC.step(
            rng,
            make_tempered_model(sampler, model, sampler.inverse_temperatures[i]),
            getsampler(sampler, i);
            init_params=init_params !== nothing ? init_params[i] : nothing,
            kwargs...
        )
        for i in 1:numtemps(sampler)
    ]

    # Make sure to collect, because we'll be using `setindex!(!)` later.
    process_to_chain = collect(1:length(sampler.inverse_temperatures))
    # Need to `copy` because this might be mutated.
    chain_to_process = copy(process_to_chain)
    state = TemperedState(
        transitions_and_states,
        sampler.inverse_temperatures,
        process_to_chain,
        chain_to_process,
        1,
        sampler.Ρ
    )

    return transition_for_chain(state), state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState;
    kwargs...
)
    if should_swap(sampler, state)
        state = swap_step(rng, model, sampler, state)
    else
        # `TemperedState` has the transitions and states in the order of
        # the processes, and performs swaps by moving the (inverse) temperatures
        # `β` between the processes, rather than moving states between processes
        # and keeping the `β` local to each process.
        # 
        # Therefore we iterate over the processes and then extract the corresponding
        # `β`, `sampler` and `state`, and take a step.
        @set! state.transitions_and_states = [
            AbstractMCMC.step(
                rng,
                make_tempered_model(sampler, model, β_for_process(state, i)),
                sampler_for_process(sampler, state, i),
                state_for_process(state, i);
                kwargs...
            )
            for i in 1:numtemps(sampler)
        ]
    end

    @set! state.total_steps += 1
    # We want to return the transition for the _first_ chain, i.e. the chain usually corresponding to `β=1.0`.
    return transition_for_chain(state), state
end


"""
    swap_step([strategy::AbstractSwapStrategy, ]rng, model, sampler, state)

Return new `state`, now with temperatures swapped according to `strategy`.

If no `strategy` is provided, the return-value of [`swapstrategy`](@ref) called on `sampler`
is used.
"""
function swap_step(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    return swap_step(swapstrategy(sampler), rng, model, sampler, state)
end

function swap_step(
    strategy::StandardSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    L = numtemps(sampler) - 1
    k = rand(rng, 1:L)
    return swap_attempt(rng, model, sampler, state, k, sampler.adapt, state.total_steps / L)
end

function swap_step(
    strategy::RandomPermutationSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    L = numtemps(sampler) - 1
    levels = Vector{Int}(undef, L)
    Random.randperm!(rng, levels)

    # Iterate through all levels and attempt swaps.
    for k in levels
        state = swap_attempt(rng, model, sampler, state, k, sampler.adapt, state.total_steps)
    end
    return state
end

function swap_step(
    strategy::NonReversibleSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    L = numtemps(sampler) - 1
    # Alternate between swapping odds and evens.
    levels = if state.total_steps % (2 * sampler.swap_every) == 0
        1:2:L
    else
        2:2:L
    end

    # Iterate through all levels and attempt swaps.
    for k in levels
        # TODO: For this swapping strategy, we should really be using the adaptation from Syed et. al. (2019),
        # but with the current one: shouldn't we at least divide `state.total_steps` by 2 since it will
        # take use two swap-attempts before we have tried swapping all of them (in expectation).
        state = swap_attempt(rng, model, sampler, state, k, sampler.adapt, state.total_steps)
    end
    return state
end
