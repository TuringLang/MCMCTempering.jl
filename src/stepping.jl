"""
    mutable struct TemperedState
        states          :: Array{Any}
        Δ               :: Vector{<:Real}
        Δ_index         :: Vector{<:Integer}
        chain_index     :: Vector{<:Integer}
        step_counter    :: Integer
        total_steps     :: Integer
        Δ_history       :: Array{<:Real, 2}
        Δ_index_history :: Array{<:Integer, 2}
        Ρ               :: Vector{AdaptiveState}
    end

A `TemperedState` struct contains the `states` of each of the parallel chains
used throughout parallel tempering as pairs of `Transition`s and `VarInfo`s,
it also stores necessary information for tempering:
- `states` is an Array of pairs of `Transition`s and `VarInfo`s, one for each 
  tempered chain
- `Δ` contains the ordered sequence of inverse temperatures
- `Δ_index` contains the current ordering to apply the temperatures to each chain, tracking swaps,
    i.e., contains the index `Δ_index[i] = j` of the temperature in `Δ`, `Δ[j]`, to apply to chain `i`
- `chain_index` contains the index `chain_index[i] = k` of the chain tempered by `Δ[i]`
    NOTE that to convert between this and `Δ_index` we simply use the `sortperm()` function
- `step_counter` maintains the number of steps taken since the last swap attempt
- `total_steps` maintains the count of the total number of steps taken
- `Δ_index_history` records the history of swaps that occur in sampling by recording the `Δ_index` at each step
- `Δ_history` records the values of the inverse temperatures, these will change if adaptation is being used
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
    states
    Δ
    Δ_index
    chain_index
    step_counter
    total_steps
    Ρ
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
    transitions_and_states = [
        AbstractMCMC.step(
            rng,
            make_tempered_model(spl, model, spl.Δ[spl.Δ_init[i]]),
            spl.internal_sampler;
            init_params=init_params !== nothing ? init_params[i] : nothing,
            kwargs...
        )
        for i in 1:length(spl.Δ)
    ]
    return (
        # Get the left-most `(transition, state)` pair, then get the `transition`.
        first(first(transitions_and_states)),
        TemperedState(
            transitions_and_states, spl.Δ, spl.Δ_init, sortperm(spl.Δ_init), 1, 1, spl.Ρ
        )
    )
end
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    spl::TemperedSampler,
    ts::TemperedState;
    kwargs...
)
    if ts.step_counter == spl.N_swap
        ts = swap_step(rng, model, spl, ts)
        @set! ts.step_counter = 0
    else
        @set! ts.states = [
            AbstractMCMC.step(
                rng,
                make_tempered_model(spl, model, ts.Δ[ts.Δ_index[i]]),
                spl.internal_sampler,
                ts.states[ts.chain_index[i]][2];
                kwargs...
            )
            for i in 1:length(ts.Δ)
        ]
        @set! ts.step_counter += 1
    end

    @set! ts.total_steps += 1
    # Use `chain_index[1]` to ensure the sample from the target is always returned for the step.
    return ts.states[ts.chain_index[1]][1], ts
end


"""
    swap_step([strategy::AbstractSwapStrategy, ]rng, model, spl, ts)

Uses the internals of the passed `TemperedSampler` - `spl` - and `TemperedState` -
`ts` - to perform a "swap step" between temperatures, in accordance with the relevant
swap strategy.
"""
function swap_step(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    ts::TemperedState
)
    return swap_step(swapstrategy(sampler), rng, model, sampler, ts)
end

function swap_step(
    strategy::StandardSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    ts::TemperedState
)
    L = length(ts.Δ) - 1
    k = rand(rng, 1:L)
    return swap_attempt(rng, model, sampler.internal_sampler, ts, k, sampler.adapt, ts.total_steps / L)
end

function swap_step(
    strategy::RandomPermutationSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    ts::TemperedState
)
    L = length(ts.Δ) - 1
    levels = Vector{Int}(undef, L)
    Random.randperm!(rng, levels)

    # Iterate through all levels and attempt swaps.
    for k in levels
        ts = swap_attempt(rng, model, sampler.internal_sampler, ts, k, sampler.adapt, ts.total_steps)
    end
    return ts
end

function swap_step(
    strategy::NonReversibleSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    ts::TemperedState
)
    L = length(ts.Δ) - 1
    # Alternate between swapping odds and evens.
    levels = if ts.total_steps % (2 * sampler.N_swap) == 0
        1:2:L
    else
        2:2:L
    end

    # Iterate through all levels and attempt swaps.
    for k in levels
        ts = swap_attempt(rng, model, sampler.internal_sampler, ts, k, sampler.adapt, ts.total_steps)
    end
    return ts
end
