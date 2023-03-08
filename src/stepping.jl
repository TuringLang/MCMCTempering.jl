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
    # Create the tempered `MultiModel`.
    multimodel = MultiModel([make_tempered_model(sampler, model, beta) for beta in state.chain_to_beta])
    # Create the tempered `MultiSampler`.
    multisampler = MultiSampler([getsampler(sampler, i) for i in 1:numtemps(sampler)])
    # Create the composition which applies `SwapSampler` first.
    sampler_composition = multisampler ∘ swapsampler(sampler)

    # Step!
    # NOTE: This will internally re-order the models according to processes before taking steps,
    # hence the resulting transitions and states will be in the order of processes, as we desire.
    transition_composition, state_composition = AbstractMCMC.step(
        rng,
        multimodel,
        sampler_composition,
        composition_state(sampler_composition, state.swapstate, state.state);
        kwargs...
    )

    # Construct the `TemperedTransition` and `TemperedState`.
    swaptransition = inner_transition(transition_composition)
    outertransition = outer_transition(transition_composition)

    swapstate = inner_state(state_composition)
    outerstate = outer_state(state_composition)

    return (
        TemperedTransition(swaptransition, outertransition),
        TemperedState(swapstate, outerstate, state.chain_to_beta)
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
