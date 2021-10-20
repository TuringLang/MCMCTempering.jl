"""
    mutable struct TemperedState
        transitions_and_states          :: Array{Any}
        Δ               :: Vector{<:Real}
        Δ_index         :: Vector{<:Integer}
        chain_index     :: Vector{<:Integer}
        step_counter    :: Integer
        total_steps     :: Integer
        Ρ               :: Vector{AdaptiveState}
    end

A `TemperedState` struct contains the `transitions_and_states` of each of the parallel chains
used throughout parallel tempering as pairs of `Transition`s and `VarInfo`s,
it also stores necessary information for tempering:
- `transitions_and_states` is a collection of `(transition, state)` pairs, one for each tempered chain.
- `Δ` contains the ordered sequence of inverse temperatures.
- `Δ_index` contains the current ordering to apply the temperatures to each chain, tracking swaps,
    i.e., contains the index `Δ_index[i] = j` of the temperature in `Δ`, `Δ[j]`, to apply to chain `i`
- `chain_index` contains the index `chain_index[i] = k` of the chain tempered by `Δ[i]`
    NOTE that to convert between this and `Δ_index` we simply use the `sortperm()` function
- `step_counter` maintains the number of steps taken since the last swap attempt
- `total_steps` maintains the count of the total number of steps taken
- `Ρ` contains all of the information required for adaptation of Δ

Example of swaps across 4 chains and the values of `chain_index` and `Δ_index`:

Chains:        chain_index:     Δ_index:
| | | |        1  2  3  4       1  2  3  4
| | | |    
 V  | |        2  1  3  4       2  1  3  4
 Λ  | |    
| | | |        2  1  3  4       2  1  3  4
| | | |    
|  V  |        2  3  1  4       3  1  2  4
|  Λ  |    
| | | |        2  3  1  4       3  1  2  4
| | | |  
"""
@concrete struct TemperedState
    transitions_and_states
    Δ
    Δ_index
    chain_index
    step_counter
    total_steps
    Ρ
end

"""
    transition_for_chain(state[, I...])

Return the transition corresponding to the chain indexed by `I...`.
If `I...` is not specified, the transition corresponding to `β=1.0` will be returned, i.e. `I = (1, )`.
"""
transition_for_chain(state::TemperedState) = transition_for_chain(state, 1)
transition_for_chain(state::TemperedState, I...) = state.transitions_and_states[state.Δ_index[I...]][1]

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
β_for_chain(state::TemperedState, I...) = state.Δ[state.Δ_index[I...]]

"""
    β_for_process(state, I...)

Return the β corresponding to the process indexed by `I...`.
"""
β_for_process(state::TemperedState, I...) = state.Δ[I...]

"""
    sampler_for_chain(sampler::TemperedSampler, state::TemperedState[, I...])

Return the sampler corresponding to the chain indexed by `I...`.
If `I...` is not specified, the sampler corresponding to `β=1.0` will be returned.
"""
sampler_for_chain(sampler::TemperedSampler, state::TemperedState) = sampler_for_chain(sampler, state, 1)
function sampler_for_chain(sampler::TemperedSampler, state::TemperedState, I...)
    return getsampler(sampler.internal_sampler, state.Δ_index[I...])
end

"""
    sampler_for_process(sampler::TemperedSampler, state::TemperedState, I...)

Return the sampler corresponding to the process indexed by `I...`.
"""
function sampler_for_process(sampler::TemperedSampler, state::TemperedState, I...)
    return getsampler(sampler.internal_sampler, I...)
end

"""
For each `β` in `Δ`, carry out a step with a tempered model at the corresponding `β` inverse temperature,
resulting in a list of transitions and states, the transition associated with `β₀ = 1` is then returned with the
rest of the information being stored in the state.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    spl::TemperedSampler;
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
            # TODO: Should we also have one a `β_for_process` for the sampler to
            # cover the initial step? Do we even _need_ this `Δ_init[1]`?
            # Can we not just assume that the `Δ` is always in the "correct" initial order?
            make_tempered_model(spl, model, spl.Δ[spl.Δ_init[i]]),
            getsampler(spl, i);
            init_params=init_params !== nothing ? init_params[i] : nothing,
            kwargs...
        )
        for i in 1:numtemps(spl)
    ]

    state = TemperedState(
        transitions_and_states, spl.Δ, copy(spl.Δ_init), sortperm(spl.Δ_init), 1, 1, spl.Ρ
    )

    return transition_for_chain(state), state
end
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    spl::TemperedSampler,
    state::TemperedState;
    kwargs...
)
    if state.step_counter == spl.N_swap
        state = swap_step(rng, model, spl, state)
        @set! state.step_counter = 0
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
                make_tempered_model(spl, model, β_for_process(state, i)),
                sampler_for_process(spl, state, i),
                state_for_process(state, i);
                kwargs...
            )
            for i in 1:numtemps(spl)
        ]
        @set! state.step_counter += 1
    end

    @set! state.total_steps += 1
    # We want to return the transition for the _first_ chain, i.e. the chain usually corresponding to `β=1.0`.
    return transition_for_chain(state), state
end


"""
    swap_step([strategy::AbstractSwapStrategy, ]rng, model, spl, state)

Uses the internals of the passed `TemperedSampler` - `spl` - and `TemperedState` -
`state` - to perform a "swap step" between temperatures, in accordance with the relevant
swap strategy.
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
    L = length(state.Δ) - 1
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
    levels = if state.total_steps % (2 * sampler.N_swap) == 0
        1:2:L
    else
        2:2:L
    end

    # Iterate through all levels and attempt swaps.
    for k in levels
        state = swap_attempt(rng, model, sampler, state, k, sampler.adapt, state.total_steps)
    end
    return state
end
