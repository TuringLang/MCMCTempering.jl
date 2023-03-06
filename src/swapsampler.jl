"""
    SwapSampler <: AbstractMCMC.AbstractSampler

# Fields
$(FIELDS)
"""
struct SwapSampler{S,O} <: AbstractMCMC.AbstractSampler
    "swap strategy to use"
    strategy::S
    "ordering assumed for input models"
    model_order::O
end

SwapSampler() = SwapSampler(ReversibleSwap())
SwapSampler(strategy) = SwapSampler(strategy, ChainOrdering())

swapstrategy(sampler::SwapSampler) = sampler.strategy
ordering(sampler::SwapSampler) = sampler.model_order

# Interaction with the state.
function model_for_chain(ordering::ProcessOrdering, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to process index, hence we map chain index to process index
    # and extract the model corresponding to said process.
    return model_for_process(ordering, sampler, model, state, chain_to_process(state, I...))
end

function model_for_chain(::ChainOrdering, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to chain index, hence we just extract the corresponding index.
    return model.models[I...]
end

function model_for_process(::ProcessOrdering, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to process index, hence we just extract the corresponding index.
    return model.models[I...]
end

function model_for_process(ordering::ChainOrdering, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to chain ordering, hence we need to map the
    # process index `I` to the chain index.
    return model_for_chain(ordering, sampler, model, state, process_to_chain(state, I...))
end

"""
    SwapTransition

Transition type for tempered samplers.
"""
@concrete struct SwapTransition
    chain_to_process
    process_to_chain
end

# NOTE: This does not have an initial `step`! This is because we need
# states to work with before we can do anything. Hence it only makes
# sense to use this sampler in composition with other samplers.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    sampler::SwapSampler,
    state::SwapState;
    kwargs...
)
    # Reset state
    @set! state.swap_acceptance_ratios = empty(state.swap_acceptance_ratios)

    # Perform a swap step.
    state = swap_step(rng, model, sampler, state)
    @set! state.total_steps += 1

    # We want to return the transition for the _first_ chain, i.e. the chain usually corresponding to `β=1.0`.
    # TODO: What should we return here?
    return SwapTransition(deepcopy(state.chain_to_process), deepcopy(state.process_to_chain)), state
end

# NOTE: The default initial `step` for `CompositionSampler` simply calls the two different
# `step` methods, but since `SwapSampler` does not have such an implementation this will fail.
# Instead we overload the initial `step` for `CompositionSampler` involving `SwapSampler` to
# first take a `step` using the non-swapsampler and then construct `SwapState` from the resulting state.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::CompositionSampler{<:AbstractMCMC.AbstractSampler,<:SwapSampler};
    kwargs...
)
    # This should hopefully be a `MultipleStates` or something since we're working with a `MultiModel`.
    state_outer_initial = last(AbstractMCMC.step(rng, model, outer_sampler(sampler); kwargs...))
    # NOTE: Since `SwapState` wraps a sequence of states from another sampler, we need `state_outer_initial`
    # to initialize the `SwapState`.
    state_inner_initial = SwapState(state_outer_initial)

    # Create the composition state, and take a full step.
    state = composition_state(sampler, state_inner_initial, state_outer_initial)
    return AbstractMCMC.step(rng, model, sampler, state; kwargs...)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::CompositionSampler{<:SwapSampler,<:AbstractMCMC.AbstractSampler};
    kwargs...
)
    # This should hopefully be a `MultipleStates` or something since we're working with a `MultiModel`.
    state_inner_initial = last(AbstractMCMC.step(rng, model, inner_sampler(sampler); kwargs...))
    # NOTE: Since `SwapState` wraps a sequence of states from another sampler, we need `state_outer_initial`
    # to initialize the `SwapState`.
    state_outer_initial = SwapState(state_inner_initial)

    # Create the composition state, and take a full step.
    state = composition_state(sampler, state_inner_initial, state_outer_initial)
    return AbstractMCMC.step(rng, model, sampler, state; kwargs...)
end

@nospecialize function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::CompositionSampler{<:SwapSampler,<:SwapSampler};
    kwargs...
)
    error("`SwapSampler` requires states from sampler other than `SwapSampler` to be initialized")
end

function swap_attempt(rng::Random.AbstractRNG, model::MultiModel, sampler::SwapSampler, state, i, j)
    # Extract the relevant transitions.
    state_i = state_for_chain(state, i)
    state_j = state_for_chain(state, j)
    # Evaluate logdensity for both parameters for each tempered density.
    # NOTE: Assumes ordering of models is according to processes.
    model_i = model_for_chain(sampler, model, state, i)
    model_j = model_for_chain(sampler, model, state, j)
    logπiθi, logπiθj = compute_logdensities(model_i, model_j, state_i, state_j)
    logπjθj, logπjθi = compute_logdensities(model_j, model_i, state_j, state_i)

    # If the proposed temperature swap is accepted according `logα`,
    # swap the temperatures for future steps.
    logα = swap_acceptance_pt(logπiθi, logπiθj, logπjθi, logπjθj)
    should_swap = -Random.randexp(rng) ≤ logα
    if should_swap
        # TODO: Rename `swap_betas!` since no betas are involved anymore?
        swap_betas!(state.chain_to_process, state.process_to_chain, i, j)
    end

    # Keep track of the (log) acceptance ratios.
    state.swap_acceptance_ratios[i] = logα

    # TODO: Handle adaptation.
    return state
end

