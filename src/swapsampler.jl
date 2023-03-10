"""
    SwapSampler <: AbstractMCMC.AbstractSampler

# Fields
$(FIELDS)
"""
struct SwapSampler{S} <: AbstractMCMC.AbstractSampler
    "swap strategy to use"
    strategy::S
end

SwapSampler() = SwapSampler(ReversibleSwap())

swapstrategy(sampler::SwapSampler) = sampler.strategy

# Interaction with the state.
# NOTE: `SwapSampler` should only every interact with `ProcessOrdering`, so we don't implement `ChainOrdering`.
function model_for_chain(ordering::ProcessOrder, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to process index, hence we map chain index to process index
    # and extract the model corresponding to said process.
    return model_for_process(ordering, sampler, model, state, chain_to_process(state, I...))
end

function model_for_process(::ProcessOrder, sampler::SwapSampler, model::MultiModel, state::SwapState, I...)
    # `model` is expected to be ordered according to process index, hence we just extract the corresponding index.
    return model.models[I...]
end

"""
    SwapTransition

Transition type for tempered samplers.
"""
@concrete struct SwapTransition
    chain_to_process
    process_to_chain
end

function composition_transition(
    sampler::CompositionSampler{<:AbstractMCMC.AbstractSampler,<:SwapSampler},
    swaptransition::SwapTransition,
    outertransition::MultipleTransitions
)
    saveall(sampler) && return CompositionTransition(outertransition, swaptransition)
    # Otherwise we have to re-order the transitions, since without the `swaptransition` there's
    # no way to recover the true ordering of the transitions.
    return MultipleTransitions(sort_by_chain(ProcessOrder(), swaptransition, outertransition.transitions))
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

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MultiModel,
    sampler::CompositionSampler{<:AbstractMCMC.AbstractSampler,<:SwapSampler},
    state;
    kwargs...
)
    # Reminder: a `swap` can be implemented in two different ways:
    #
    # 1. Swap the models and leave ordering of (sampler, state)-pair unchanged.
    # 2. Swap (sampler, state)-pairs and leave ordering of models unchanged.
    #
    # (1) has the properties:
    # + Easy to keep `outerstate` and `swapstate` in sync since their ordering is never changed.
    # - Ordering of `outerstate` no longer corresponds to ordering of models, i.e. the returned
    #   `outerstate.states[i]` does no longer correspond to a state targeting `model.models[i]`.
    #   This will have to be adjusted in the `AbstractMCMC.bundle_samples` before, say, converting
    #   into a `MCMCChains.Chains`.
    #
    # (2) has the properties:
    # + Returned `outertransition` (and `outerstate`, if we want) has the same ordering as the models,
    #   i.e. `outerstate.states[i]` now corresponds to `model.models[i]`!
    # - Need to keep `outerstate` and `swapstate` in sync since their ordering now changes.
    # - Need to also re-order `outersampler.samplers` :/
    #
    # Here (as in, below) we go with option (1), i.e. re-order the `models`.
    # A full `step` then is as follows:
    # 1. Sort models according to index processes using the `swapstate` from previous iteration.
    # 2. Take step with `swapsampler`.
    # 3. Sort models _again_ according to index processes using the new `swapstate`, since we
    #    might have made a swap in (2).
    # 4. Run multi-sampler.

    outersampler, swapsampler = outer_sampler(sampler), inner_sampler(sampler)

    # Get the states.
    outerstate_prev, swapstate_prev = outer_state(state), inner_state(state)

    # Re-order the models.
    chain2models = model.models  # but keep the original chain → model around because we'll re-order again later
    @set! model.models = models_by_processes(ChainOrder(), chain2models, swapstate_prev)

    # Step for the swap-sampler.
    swaptransition, swapstate = AbstractMCMC.step(
        rng, model, swapsampler, state_from(model, swapstate_prev, outerstate_prev);
        kwargs...
    )

    # Re-order the models AGAIN, since we might have swapped some.
    @set! model.models = models_by_processes(ChainOrder(), chain2models, swapstate)

    # Create the current state from `outerstate_prev` and `swapstate`, and `step` for `outersampler`.`
    outertransition, outerstate = AbstractMCMC.step(
        # HACK: We really need the `state_from` here despite the fact that `SwapSampler` does note
        # change the `swapstates.states` itself, but we might require a re-computation of certain
        # quantities from the `model`, which has now potentially been re-ordered (see above).
        # NOTE: We do NOT do `state_from(model, outerstate_prev, swapstate)` because as of now,
        # `swapstate` does not implement `getparams_and_logprob`.
        rng, model, outersampler, state_from(model, outerstate_prev, outerstate_prev);
        kwargs...
    )

    # TODO: Should we re-order the transitions?
    # Currently, one has to re-order the `outertransition` according to `swaptransition`
    # in the `bundle_samples`. Is this the right approach though?
    # TODO: We should at least re-order transitions in the case where `saveall(sampler) == false`!
    # In this case, we'll just return the transition without the swap-transition, hence making it
    # impossible to reconstruct the actual ordering!
    return (
        composition_transition(sampler, swaptransition, outertransition),
        composition_state(sampler, swapstate, outerstate)
    )
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

"""
    swap_attempt(rng, model, sampler, state, i, j)

Attempt to swap the temperatures of two chains by tempering the densities and
calculating the swap acceptance ratio; then swapping if it is accepted.
"""
function swap_attempt(rng::Random.AbstractRNG, model::MultiModel, sampler::SwapSampler, state, i, j)
    # Extract the relevant transitions.
    state_i = state_for_chain(state, i)
    state_j = state_for_chain(state, j)
    # Evaluate logdensity for both parameters for each tempered density.
    # NOTE: `SwapSampler` should only be working with models ordered according to `ProcessOrder`,
    # never `ChainOrder`, hence why we have the below.
    model_i = model_for_chain(ProcessOrder(), sampler, model, state, i)
    model_j = model_for_chain(ProcessOrder(), sampler, model, state, j)
    logπiθi, logπiθj = compute_logdensities(model_i, model_j, state_i, state_j)
    logπjθj, logπjθi = compute_logdensities(model_j, model_i, state_j, state_i)

    # If the proposed temperature swap is accepted according `logα`,
    # swap the temperatures for future steps.
    logα = swap_acceptance_pt(logπiθi, logπiθj, logπjθi, logπjθj)
    should_swap = -Random.randexp(rng) ≤ logα
    if should_swap
        swap!(state.chain_to_process, state.process_to_chain, i, j)
    end

    # Keep track of the (log) acceptance ratios.
    state.swap_acceptance_ratios[i] = logα

    # TODO: Handle adaptation.
    return state
end
