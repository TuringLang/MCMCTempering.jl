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
    # `state`'s `internal[i]` contains the transition and state of the `i`th
    # chain, `state` also contains a map of each chain's current inverse
    # temperature `β`. Here, we initialise each chain using the corresponding 
    # `β`-tempered `model`, `sampler` and `init_params`.
    internal = [
        AbstractMCMC.step(
            rng,
            make_tempered_model(model, sampler.inverse_temperatures[i]),
            get_sampler(sampler, i);
            init_params=get_init_params(init_params, i),
            kwargs...
        )
        for i in 1:numtemps(sampler)
    ]
    state = init_state(internal, sampler)

    # Burn-in _without_ attempting any swaps, this is different to the `discard_initial`
    # process exposed via `AbstractMCMC.sample`
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
    return get_fresh_transition(state), state
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
    @set! state.is_swap = [false for _ in 1:numtemps(sampler)]

    if should_swap(sampler, state)
        state = swap(rng, model, sampler, state)
    end

    state = step(rng, model, sampler, state; kwargs...)
    @set! state.total_steps += 1

    return get_fresh_transition(state), state
end

function step(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState;
    kwargs...
)
    # Record all transitions prior to stepping
    prev_transitions = [deepcopy(get_transition(state, i)) for i in 1:numtemps(sampler)]

    # `state`'s `internal[i]` contains the transition and state of the `i`th
    # chain, `state` also contains a map of each chain's current inverse temperature `β`.
    # We iterate over the chains, stepping using the corresponding `β`-tempered `model`,
    # `sampler` and `state`.
    @set! state.internal = [
        AbstractMCMC.step(
            rng,
            make_tempered_model(model, get_inverse_temperature(state, i)),
            get_sampler(sampler, i),
            get_state(state, i);
            kwargs...
        )
        for i in 1:numtemps(sampler)
    ]

    # After stepping, extract the new transitions to compare to the previous ones. If a
    # swap has taken place involving the `i`th chain or that chain is stale then we check
    # that there has been a succesful proposal under the new inverse temperature. If there
    # isn't, i.e. previous transition == new transition, then mark the chain as stale such
    # that we do not record samples from it until a successful proposal is made.
    for i in 1:numtemps(sampler)
        if state.is_stale[i] || state.is_swap[i]
            if prev_transitions[i] == get_transition(state, i)
                @set! state.is_stale[i] = true
            else
                @set! state.is_stale[i] = false
            end
        end
    end
    return state
end

"""
    swap([strategy::AbstractSwapStrategy, ]rng, model, sampler, state)

Return a new `state`, now with temperatures swapped according to `strategy`.

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
        state = swap_attempt(rng, model, state, k, k + 1)
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
    # Alternate between attempting to swap chains corresponding
    # to odd and even indices of the temperature ladder
    odd = state.total_steps % (2 * sampler.swap_every) != 0
    for k in [Int(2 * i - odd) for i in 1:(floor((numtemps(sampler) - 1 + odd) / 2))]
        state = swap_attempt(rng, model, state, k, k + 1)
    end
    return state
end

function swap(
    strategy::SingleSwap,
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState
)
    # Select one index `k` of the temperature ladder and
    # swap the corresponding chain and its neighbour
    k = rand(rng, 1:(numtemps(sampler) - 1))
    return swap_attempt(rng, model, state, k, k + 1)
end

function swap(
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
    return swap_attempt(rng, model, state, i, j)
end

function swap(
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
        state = swap_attempt(rng, model, state, i, j)
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
