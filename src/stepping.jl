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


"""
For each `β` in `Δ`, carry out a step with a tempered model at the corresponding `β` inverse temperature,
resulting in a list of transitions and states, the transition associated with `β₀ = 1` is then returned with the
rest of the information being stored in the state.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    spl::TemperedSampler;
    kwargs...
)
    states = [
        AbstractMCMC.step(
            rng,
            make_tempered_model(model, spl.Δ[spl.Δ_init[i]]),
            spl.internal_sampler;
            kwargs...
        )
        for i in 1:length(spl.Δ)
    ]
    return (
        states[sortperm(spl.Δ_init)[1]][1],
        TemperedState(
            states,spl.Δ, spl.Δ_init, sortperm(spl.Δ_init), 1, 1, Array{Real, 2}(spl.Δ'), Array{Integer, 2}(spl.Δ_init'), spl.Ρ
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
        ts.step_counter = 0
    else
        ts.states = [
            AbstractMCMC.step(
                rng,
                make_tempered_model(model, ts.Δ[ts.Δ_index[i]]),
                spl.internal_sampler,
                ts.states[i][2];
                kwargs...
            )
            for i in 1:length(ts.Δ)
        ]
        ts.step_counter += 1
    end

    ts.Δ_history = vcat(ts.Δ_history, Array{Real, 2}(ts.Δ'))
    ts.Δ_index_history = vcat(ts.Δ_index_history, Array{Integer, 2}(ts.Δ_index'))
    ts.total_steps += 1
    return ts.states[ts.chain_index[1]][1], ts  # Use chain_index[1] to ensure the sample from the target is always returned for the step
end


"""
    swap_step(rng, model, spl, ts)

Uses the internals of the passed `TemperedSampler` - `spl` - and `TemperedState` -
`ts` - to perform a "swap step" between temperatures, in accordance with the relevant
swap strategy.
"""
function swap_step(
    rng::Random.AbstractRNG,
    model,
    spl::TemperedSampler,
    ts::TemperedState
)
    L = length(ts.Δ) - 1
    sampler = spl.internal_sampler

    if spl.swap_strategy == :standard

        k = rand(rng, Distributions.Categorical(L))  # Pick randomly from 1, 2, ..., k - 1
        ts = swap_attempt(model, sampler, ts, k, spl.adapt, ts.total_steps / L)

    else

        # Define a vector to populate with levels at which to propose swaps according to swap_strategy
        levels = Vector{Int}(undef, L)
        if spl.swap_strategy == :nonrev
            if ts.step_counter % (2 * spl.N_swap) == 0
                levels = 1:2:L
            else
                levels = 2:2:L
            end
        elseif spl.swap_strategy == :randperm
            randperm!(rng, levels)
        end

        for k in levels
            ts = swap_attempt(model, sampler, ts, k, spl.adapt, ts.total_steps)
        end

    end
    return ts
end
