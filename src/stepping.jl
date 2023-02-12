"""
    should_swap(sampler, state)

Return `true` if a swap should happen at this iteration, and `false` otherwise.
"""
function should_swap(sampler::TemperedSampler, state::TemperedState)
    return state.total_steps % sampler.swap_every == 0
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
    internal_state = [
        AbstractMCMC.step(
            rng,
            make_tempered_model(model, sampler.inverse_temperatures[i]),
            get_sampler(sampler, i);
            init_params=get_init_params(init_params, i),
            kwargs...
        )
        for i in 1:numtemps(sampler)
    ]
    state = init_state(internal_state, collect(1:numtemps(sampler)), sampler)

    if N_burnin > 0
        AbstractMCMC.@ifwithprogresslogger burnin_progress name = "Burn-in" begin
            if burnin_progress
                threshold = N_burnin ÷ 200
                next_update = threshold
            end
            for i in 1:N_burnin
                if burnin_progress && i >= next_update
                    ProgressLogging.@logprogress i / N_burnin
                    next_update = i + threshold
                end
                state = step(rng, model, sampler, state; kwargs...)
                @set! state.burnin_steps += 1
            end
        end
    end
    return get_transition(state), state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState;
    kwargs...
)
    # Reset
    @set! state.swap_acceptance_ratios = empty(state.swap_acceptance_ratios)
    @set! state.is_swap = false

    if should_swap(sampler, state)
        state = swap(rng, model, sampler, state)
        @set! state.is_swap = true
    end

    state = step(rng, model, sampler, state; kwargs...)
    @set! state.total_steps += 1

    # We want to return the transition for the chain corresponding to `β = 1.0`
    return get_transition(state), state
end

function step(
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
    @set! state.internal_state = [
        AbstractMCMC.step(
            rng,
            make_tempered_model(model, get_inverse_temperature(state, i)),
            get_sampler(sampler, i),
            get_state(state, i);
            kwargs...
        )
        for i in 1:numtemps(sampler)
    ]
    return state
end

"""
    swap([strategy::AbstractSwapStrategy, ]rng, model, sampler, state)

Return new `state`, now with temperatures swapped according to `strategy`.

If no `strategy` is provided, the return-value of [`swapstrategy`](@ref) called on `sampler`
is used.
"""
function swap(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    return swap(swapstrategy(sampler), rng, model, sampler, state)
end

function swap(
    strategy::StandardSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    L = numtemps(sampler) - 1
    k = rand(rng, 1:L)
    return swap_attempt(rng, model, state, k)
end

function swap(
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
        state = swap_attempt(rng, model, state, k)
    end
    return state
end

function swap(
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
        state = swap_attempt(rng, model, state, k)
    end
    return state
end

function swap(
    strategy::NoSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    return state
end
