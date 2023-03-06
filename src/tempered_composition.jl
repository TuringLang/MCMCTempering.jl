Base.@kwdef struct TemperedComposition{SplT,A,SwapT,Adapt} <: AbstractMCMC.AbstractSampler
    "sampler(s) used to target the tempered distributions"
    sampler::SplT
    "collection of inverse temperatures β; β[i] correponds i-th tempered model"
    chain_to_beta::A
    "strategy to use for swapping"
    swapstrategy::SwapT=ReversibleSwap()
    # TODO: Remove `adapt` and just consider `adaptation_states=nothing` as no adaptation.
    "boolean flag specifying whether or not to adapt"
    adapt=false
    "adaptation parameters"
    adaptation_states::Adapt=nothing
end

TemperedComposition(sampler, chain_to_beta) = TemperedComposition(; sampler, chain_to_beta)

numtemps(sampler::TemperedComposition) = length(sampler.chain_to_beta)

getsampler(sampler::TemperedComposition, I...) = getsampler(sampler.sampler, I...)

swapsampler(sampler::TemperedComposition) = SwapSampler(sampler.swapstrategy, ProcessOrdering())

# Simple wrapper state which also contains the temperatures.
@concrete struct TemperState
    swapstate
    state
    chain_to_beta
end

inner_state(state::TemperState) = state.swapstate
outer_state(state::TemperState) = state.state

state_for_process(state::TemperState, I...) = state_for_process(state.swapstate, I...)

beta_for_chain(state::TemperState, I...) = state.chain_to_beta[I...]
beta_for_process(state::TemperState, I...) = state.chain_to_beta[process_to_chain(state.swapstate, I...)]

function model_for_process(sampler::TemperedComposition, model, state::TemperState, I...)
    return make_tempered_model(sampler, model, beta_for_process(state, I...))
end

function sampler_for_process(sampler::TemperedComposition, state::TemperState, I...)
    return _sampler_for_process_temper(sampler.sampler, state.swapstate, I...)
end

# If `sampler` is a `MultiSampler`, we assume it's ordered according to chains.
_sampler_for_process_temper(sampler::MultiSampler, state, I...) = sampler.samplers[process_to_chain(state, I...)]
# Otherwise, we just use the same sampler for everything.
_sampler_for_process_temper(sampler, state, I...) = sampler

@concrete struct TemperTransition
    swaptransition
    transition
end

function transition_for_chain(transition::TemperTransition, I...)
    chain_idx = transition.swaptransition.chain_to_process[I...]
    return transition.transition.transitions[chain_idx]
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedComposition;
    kwargs...
)
    # Create a `MultiSampler` and `MultiModel`.
    multimodel = MultiModel([
        make_tempered_model(sampler, model, sampler.chain_to_beta[i])
        for i in 1:numtemps(sampler)
    ])
    multisampler = MultiSampler([getsampler(sampler, i) for i in 1:numtemps(sampler)])
    multistate = last(AbstractMCMC.step(rng, multimodel, multisampler; kwargs...))

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

    return AbstractMCMC.step(rng, model, sampler, TemperState(swapstate, multistate, sampler.chain_to_beta))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedComposition,
    state::TemperState;
    kwargs...
)
    # Get the samplers.
    swapspl = swapsampler(sampler)
    # Extract the previous states.
    swapstate_prev, multistate_prev = inner_state(state), outer_state(state)

    # BUT to call `make_tempered_model`, the temperatures need to be available. 
    multimodel_swap = MultiModel([model_for_process(sampler, model, state, i) for i in 1:numtemps(sampler)])

    # Update the `swapstate`.
    swapstate = state_from(model, swapstate_prev, multistate_prev)
    # Take a step with the swap sampler.
    swaptransition, swapstate = AbstractMCMC.step(rng, multimodel_swap, swapspl, swapstate; kwargs...)

    # Update `state`.
    @set! state.swapstate = swapstate
    
    # Create the multi-versions with the ordering corresponding to the processes. This way, whenever we make
    # use of `Threads.@threads` or the like, we get the same ordering.
    # NOTE: If the user-provided `model` is a `MultiModel`, then `model_for_process` implementation
    # for `TemperedComposition` will assume the models are ordered according to chains rather than processes.
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
        TemperTransition(swaptransition, multitransition),
        TemperState(swapstate, multistate, state.chain_to_beta)
    )
end

