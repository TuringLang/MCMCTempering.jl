"""
    should_swap(sampler, state)

Return `true` if a swap should happen at this iteration, and `false` otherwise.
"""
function should_swap(sampler::TemperedSampler, state::TemperedState)
    return state.total_steps % sampler.swap_every == 1
end

get_init_params(x, _)= x
get_init_params(init_params::Nothing, _) = nothing
get_init_params(init_params::AbstractVector{<:Real}, _) = copy(init_params)
get_init_params(init_params::AbstractVector{<:AbstractVector{<:Real}}, i) = init_params[i]

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler;
    N_burnin::Integer=0,
    burnin_progress::Bool=AbstractMCMC.PROGRESS[],
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
            init_params=get_init_params(init_params, i),
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
        0,
        sampler.adaptation_states,
        false,
        Dict{Int,Float64}()
    )

    if N_burnin > 0
        AbstractMCMC.@ifwithprogresslogger burnin_progress name = "Burn-in" begin
            # Determine threshold values for progress logging
            # (one update per 0.5% of progress)
            if burnin_progress
                threshold = N_burnin ÷ 200
                next_update = threshold
            end

            for i in 1:N_burnin
                if burnin_progress && i >= next_update
                    ProgressLogging.@logprogress i / N_burnin
                    next_update = i + threshold
                end
                state = no_swap_step(rng, model, sampler, state; kwargs...)
                @set! state.burnin_steps += 1
            end
        end
    end

    return transition_for_chain(state), state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState;
    kwargs...
)
    # Reset state
    @set! state.swap_acceptance_ratios = empty(state.swap_acceptance_ratios)

    if should_swap(sampler, state)
        state = swap_step(rng, model, sampler, state)
        @set! state.is_swap = true
    else
        state = no_swap_step(rng, model, sampler, state; kwargs...)
        @set! state.is_swap = false
    end

    @set! state.total_steps += 1

    # We want to return the transition for the _first_ chain, i.e. the chain usually corresponding to `β=1.0`.
    return transition_for_chain(state), state
end

function no_swap_step(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState;
    kwargs...
)
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
            make_tempered_model(sampler, model, beta_for_process(state, i)),
            sampler_for_process(sampler, state, i),
            state_for_process(state, i);
            kwargs...
        )
        for i in 1:numtemps(sampler)
    ]

    return state
end

"""
    swap_step([strategy::AbstractSwapStrategy, ]rng, model, sampler, state)

Return a new `state`, with temperatures possibly swapped according to `strategy`.

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
    strategy::ReversibleSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    # Randomly select whether to attempt swaps between chains
    # corresponding to odd or even indices of the temperature ladder
    odd = rand([true, false])
    for k in [Int(2 * i - odd) for i in 1:(floor((numtemps(sampler) - 1 + odd) / 2))]
        state = swap_attempt(rng, model, sampler, state, k, k + 1, sampler.adapt)
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
    # Alternate between attempting to swap chains corresponding
    # to odd and even indices of the temperature ladder
    odd = state.total_steps % (2 * sampler.swap_every) != 0
    for k in [Int(2 * i - odd) for i in 1:(floor((numtemps(sampler) - 1 + odd) / 2))]
        state = swap_attempt(rng, model, sampler, state, k, k + 1, sampler.adapt)
    end
    return state
end

function swap_step(
    strategy::SingleSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    # Randomly pick one index `k` of the temperature ladder and
    # attempt a swap between the corresponding chain and its neighbour
    k = rand(rng, 1:(numtemps(sampler) - 1))
    return swap_attempt(rng, model, sampler, state, k, k + 1, sampler.adapt)
end

function swap_step(
    strategy::SingleRandomSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    # Randomly pick two temperature ladder indices in order to
    # attempt a swap between the corresponding chains
    chains = Set(1:numtemps(sampler))
    i = pop!(chains, rand(rng, chains))
    j = pop!(chains, rand(rng, chains))
    return swap_attempt(rng, model, sampler, state, i, j, sampler.adapt)
end

function swap_step(
    strategy::RandomSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    # Iterate through all of temperature ladder indices, picking random
    # pairs and attempting swaps between the corresponding chains
    chains = Set(1:numtemps(sampler))
    while length(chains) >= 2
        i = pop!(chains, rand(rng, chains))
        j = pop!(chains, rand(rng, chains))
        state = swap_attempt(rng, model, sampler, state, i, j, sampler.adapt)
    end
    return state
end

function swap_step(
    strategy::NoSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    return state
end