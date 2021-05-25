"""
    mutable struct TemperedState
        states        :: Array{Any}
        Δ             :: Vector{<:AbstractFloat}
        Δ_state       :: Vector{<:Integer}
        step_counter  :: Integer
        swap_history  :: Array{<:Integer, 2}
    end

A `TemperedState` struct contains the `states` of each of the parallel chains used throughout parallel tempering,
as pairs of `Transition`s and `VarInfo`s, it also stores necessary information for tempering:
- `states` is an Array of pairs of `Transition`s and `VarInfo`s, one for each tempered chain
- `Δ_state` contains the current ordering of temperatures to apply to the chains, i.e. indices to call the temperature ladder with
- `step_counter` maintains the number of steps taken since the last swap attempt
- `swap_history` reccords the history of swaps that occur in sampling
"""
mutable struct TemperedState
    states        :: Array{Any}
    Δ             :: Vector{<:AbstractFloat}
    Δ_state       :: Vector{<:Integer}
    step_counter  :: Integer
    swap_history  :: Array{<:Integer, 2}
end


"""
For each `β` in `Δ`, carry out a step with a tempered model at the corresponding `β` inverse temperature,
resulting in a list of transitions and states, the first transition is then returned with the
rest of the information being stored in the state.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    spl::TemperedSampler,
    kwargs...
)
    states = [
        AbstractMCMC.step(
            rng,
            make_tempered_model(model, spl.Δ[Δi]),
            spl.internal_sampler,
            kwargs...
        )
        for Δi in spl.Δ_init
    ]
    return states[1][1], TemperedState(states, spl.Δ, spl.Δ_init, 1, Array{Integer, 2}(spl.Δ_init'))
end


function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    spl::TemperedSampler,
    ts::TemperedState,
    kwargs...
)
    if ts.step_counter == spl.N_swap

        ts.Δ_state = swap_step(rng, model, spl, ts)
        ts.step_counter = 0

    else

        ts.states = [
            AbstractMCMC.step(
                rng,
                make_tempered_model(model, ts.Δ[ts.Δ_state[i]]),
                spl.internal_sampler,
                ts.states[i][2];
                kwargs...
            )
            for i in 1:length(ts.Δ)
        ]
        ts.step_counter += 1
    end

    ts.swap_history = vcat(ts.swap_history, Array{Integer, 2}(ts.Δ_state'))
    
    return ts.states[1][1], ts
end


function swap_step(
    rng::Random.AbstractRNG,
    model,
    spl::TemperedSampler,
    ts::TemperedState
)
    L = length(spl.Δ) - 1
    sampler = spl.internal_sampler

    if spl.swap_strategy == :standard

        k = rand(rng, Distributions.Categorical(L)) # Pick randomly from 1, 2, ..., k-1
        Δ_state = swap_attempt(model, sampler, ts.states, k, spl.Δ, ts.Δ_state)
    
    else

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
            Δ_state = swap_attempt(model, sampler, ts.states, k, spl.Δ, ts.Δ_state)
        end

    end

    return Δ_state
end
