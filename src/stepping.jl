"""
    should_swap(sampler, state)

Return `true` if a swap should happen at this iteration, and `false` otherwise.
"""
function should_swap(sampler::TemperedSampler, state::TemperedState)
    return state.total_steps % sampler.swap_every == 1
end

get_init_params(x, _) = x
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

    # Create a `MultiSampler` and `MultiModel`.
    multimodel = MultiModel(
        make_tempered_model(sampler, model, sampler.inverse_temperatures[i])
        for i in 1:numtemps(sampler)
    )
    multisampler = MultiSampler(getsampler(sampler, i) for i in 1:numtemps(sampler))
    multitransition, multistate = AbstractMCMC.step(
        rng, multimodel, multisampler;
        init_params=init_params,
        kwargs...
    )

    # Make sure to collect, because we'll be using `setindex!(!)` later.
    process_to_chain = collect(1:length(sampler.inverse_temperatures))
    # Need to `copy` because this might be mutated.
    chain_to_process = copy(process_to_chain)
    state = TemperedState(
        multitransition.transitions,
        multistate.states,
        sampler.inverse_temperatures,
        process_to_chain,
        chain_to_process,
        1,
        0,
        sampler.adaptation_states,
        false,
        Dict{Int,Float64}()
    )

    # TODO: Move this to AbstractMCMC. Or better, add to AbstractMCMC a way to
    # specify a callback to be used for the `discard_initial`.
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

    return TemperedTransition(transition_for_chain(state)), state
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

    isswap = should_swap(sampler, state)
    if isswap
        state = swap_step(rng, model, sampler, state)
        @set! state.is_swap = true
    else
        state = no_swap_step(rng, model, sampler, state; kwargs...)
        @set! state.is_swap = false
    end

    @set! state.total_steps += 1

    # We want to return the transition for the _first_ chain, i.e. the chain usually corresponding to `β=1.0`.
    return TemperedTransition(transition_for_chain(state), isswap), state
end

function no_swap_step(
    rng::Random.AbstractRNG,
    model,
    sampler::TemperedSampler,
    state::TemperedState;
    kwargs...
)
    # Create the multi-versions with the ordering corresponding to the processes.
    multimodel = MultiModel(model_for_process(sampler, model, state, i) for i in 1:numtemps(sampler))
    multisampler = MultiSampler(sampler_for_process(sampler, state, i) for i in 1:numtemps(sampler))
    multistate = MultipleStates(state_for_process(state, i) for i in 1:numtemps(sampler))

    # And then step.
    multitransition, multistate_next = AbstractMCMC.step(
        rng,
        multimodel,
        multisampler,
        multistate;
        kwargs...
    )

    # Update the `TemperedState`.
    @set! state.transitions = multitransition.transitions
    @set! state.states = multistate_next.states

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
    sampler,
    state
)
    return swap_step(swapstrategy(sampler), rng, model, sampler, state)
end

function swap_step(
    strategy::ReversibleSwap,
    rng::Random.AbstractRNG,
    model,
    sampler,
    state
)
    # Randomly select whether to attempt swaps between chains
    # corresponding to odd or even indices of the temperature ladder
    odd = rand(rng, Bool)
    for k in [Int(2 * i - odd) for i in 1:(floor((numtemps(sampler) - 1 + odd) / 2))]
        state = swap_attempt(rng, model, sampler, state, k, k + 1)
    end
    return state
end

function swap_step(
    strategy::ReversibleSwap,
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler,
    state
)
    # Randomly select whether to attempt swaps between chains
    # corresponding to odd or even indices of the temperature ladder
    odd = rand(rng, Bool)
    for k in [Int(2 * i - odd) for i in 1:(floor((length(model.models) - 1 + odd) / 2))]
        state = swap_attempt(rng, model, sampler, state, k, k + 1)
    end
    return state
end


function swap_step(
    strategy::NonReversibleSwap,
    rng::Random.AbstractRNG,
    model,
    sampler,
    state
)
    # Alternate between attempting to swap chains corresponding
    # to odd and even indices of the temperature ladder
    odd = state.total_steps % (2 * sampler.swap_every) != 0
    for k in [Int(2 * i - odd) for i in 1:(floor((numtemps(sampler) - 1 + odd) / 2))]
        state = swap_attempt(rng, model, sampler, state, k, k + 1)
    end
    return state
end

function swap_step(
    strategy::SingleSwap,
    rng::Random.AbstractRNG,
    model,
    sampler,
    state
)
    # Randomly pick one index `k` of the temperature ladder and
    # attempt a swap between the corresponding chain and its neighbour
    k = rand(rng, 1:(numtemps(sampler) - 1))
    return swap_attempt(rng, model, sampler, state, k, k + 1)
end

function swap_step(
    strategy::SingleRandomSwap,
    rng::Random.AbstractRNG,
    model,
    sampler,
    state
)
    # Randomly pick two temperature ladder indices in order to
    # attempt a swap between the corresponding chains
    chains = Set(1:numtemps(sampler))
    i = pop!(chains, rand(rng, chains))
    j = pop!(chains, rand(rng, chains))
    return swap_attempt(rng, model, sampler, state, i, j)
end

function swap_step(
    strategy::RandomSwap,
    rng::Random.AbstractRNG,
    model,
    sampler,
    state
)
    # Iterate through all of temperature ladder indices, picking random
    # pairs and attempting swaps between the corresponding chains
    chains = Set(1:numtemps(sampler))
    while length(chains) >= 2
        i = pop!(chains, rand(rng, chains))
        j = pop!(chains, rand(rng, chains))
        state = swap_attempt(rng, model, sampler, state, i, j)
    end
    return state
end

function swap_step(
    strategy::NoSwap,
    rng::Random.AbstractRNG,
    model,
    sampler,
    state
)
    return state
end
