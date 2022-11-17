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
    # Reset.
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
            make_tempered_model(sampler, model, β_for_process(state, i)),
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
