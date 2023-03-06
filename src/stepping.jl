get_init_params(x, _) = x
get_init_params(init_params::Nothing, _) = nothing
get_init_params(init_params::AbstractVector{<:Real}, _) = copy(init_params)
get_init_params(init_params::AbstractVector{<:AbstractVector{<:Real}}, i) = init_params[i]

@concrete struct TemperedTransition
    swaptransition
    transition
end

function transition_for_chain(transition::TemperedTransition, I...)
    chain_idx = transition.swaptransition.chain_to_process[I...]
    return transition.transition.transitions[chain_idx]
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedSampler;
    N_burnin::Integer=0,
    burnin_progress::Bool=AbstractMCMC.PROGRESS[],
    kwargs...
)
    # Create a `MultiSampler` and `MultiModel`.
    multimodel = MultiModel([
        make_tempered_model(sampler, model, sampler.chain_to_beta[i])
        for i in 1:numtemps(sampler)
    ])
    multisampler = MultiSampler([getsampler(sampler, i) for i in 1:numtemps(sampler)])
    multistate = last(AbstractMCMC.step(rng, multimodel, multisampler; kwargs...))

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
                multistate = last(AbstractMCMC.step(rng, multimodel, multisampler, multistate; kwargs...))
            end
        end
    end

    # Make sure to collect, because we'll be using `setindex!(!)` later.
    process_to_chain = collect(1:length(sampler.chain_to_beta))
    # Need to `copy` because this might be mutated.
    chain_to_process = copy(process_to_chain)
    swapstate = SwapState(
        multistate.states,
        chain_to_process,
        process_to_chain,
        1,
        Dict{Int,Float64}(),
    )

    return AbstractMCMC.step(rng, model, sampler, TemperedState(swapstate, multistate, sampler.chain_to_beta))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedSampler,
    state::TemperedState;
    kwargs...
)
    # Get the samplers.
    swapspl = swapsampler(sampler)
    # Extract the previous states.
    swapstate_prev, multistate_prev = state.swapstate, state.state

    # BUT to call `make_tempered_model`, the temperatures need to be available. 
    multimodel_swap = MultiModel([model_for_process(sampler, model, state, i) for i in 1:numtemps(sampler)])

    # Update the `swapstate`.
    swapstate = state_from(model, swapstate_prev, multistate_prev)
    # Take a step with the swap sampler.
    swaptransition, swapstate = AbstractMCMC.step(rng, multimodel_swap, swapspl, swapstate; kwargs...)

    # Update `swapstate` in `state`.
    @set! state.swapstate = swapstate
    
    # Create the multi-versions with the ordering corresponding to the processes.
    # NOTE: If the user-provided `model` is a `MultiModel`, then `model_for_process` implementation
    # for `TemperedSampler` will assume the models are ordered according to chains rather than processes.
    multimodel = MultiModel([model_for_process(sampler, model, state, i) for i in 1:numtemps(sampler)])
    # NOTE: If `sampler.sampler` is a `MultiSampler`, then we should just select the corresponding index.
    # Otherwise, we just replicate the `sampler.sampler`.
    multispl = MultiSampler([sampler_for_process(sampler, state, i) for i in 1:numtemps(sampler)])
    # A `SwapState` has to contain the states for the other sampler, otherwise the `SwapSampler` won't be
    # able to compute the logdensities, etc.
    multistate = MultipleStates([state_for_process(state, i) for i in 1:numtemps(sampler)])

    # Take a step with the multi sampler.
    multitransition, multistate = AbstractMCMC.step(rng, multimodel, multispl, multistate; kwargs...)

    # TODO: Should we still call `composition_transition`?
    return (
        TemperedTransition(swaptransition, multitransition),
        TemperedState(swapstate, multistate, state.chain_to_beta)
    )
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
    model::MultiModel,
    sampler,
    state
)
    # Randomly select whether to attempt swaps between chains
    # corresponding to odd or even indices of the temperature ladder
    odd = rand(rng, Bool)
    # TODO: Use integer-division.
    for k in [Int(2 * i - odd) for i in 1:(floor((length(model) - 1 + odd) / 2))]
        state = swap_attempt(rng, model, sampler, state, k, k + 1)
    end
    return state
end


function swap_step(
    strategy::NonReversibleSwap,
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler,
    state::SwapState  # we're accessing `total_steps` restrict the type here
)
    # Alternate between attempting to swap chains corresponding
    # to odd and even indices of the temperature ladder
    odd = state.total_steps % 2 != 0
    # TODO: Use integer-division.
    for k in [Int(2 * i - odd) for i in 1:(floor((length(model) - 1 + odd) / 2))]
        state = swap_attempt(rng, model, sampler, state, k, k + 1)
    end
    return state
end

function swap_step(
    strategy::SingleSwap,
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler,
    state
)
    # Randomly pick one index `k` of the temperature ladder and
    # attempt a swap between the corresponding chain and its neighbour
    k = rand(rng, 1:(length(model) - 1))
    return swap_attempt(rng, model, sampler, state, k, k + 1)
end

function swap_step(
    strategy::SingleRandomSwap,
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler,
    state
)
    # Randomly pick two temperature ladder indices in order to
    # attempt a swap between the corresponding chains
    chains = Set(1:length(model))
    i = pop!(chains, rand(rng, chains))
    j = pop!(chains, rand(rng, chains))
    return swap_attempt(rng, model, sampler, state, i, j)
end

function swap_step(
    strategy::RandomSwap,
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler,
    state
)
    # Iterate through all of temperature ladder indices, picking random
    # pairs and attempting swaps between the corresponding chains
    chains = Set(1:length(model))
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
