"""
    TemperedState

A state for a tempered sampler.

# Fields
$(FIELDS)
"""
@concrete struct TemperedState
    "state for swap-sampler"
    swapstate
    "state for the main sampler"
    state
    "inverse temperature for each of the chains"
    chain_to_beta
end

"""
    TemperedTransition

A transition for a tempered sampler.

# Fields
$(FIELDS)
"""
@concrete struct TemperedTransition
    "transition for swap-sampler"
    swaptransition
    "transition for the main sampler"
    transition
end

"""
    TemperedSampler <: AbstractMCMC.AbstractSampler

A `TemperedSampler` struct wraps a sampler upon which to apply the Parallel Tempering algorithm.

# Fields

$(FIELDS)
"""
Base.@kwdef struct TemperedSampler{SplT,A,SwapT,Adapt} <: AbstractMCMC.AbstractSampler
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

TemperedSampler(sampler, chain_to_beta; kwargs...) = TemperedSampler(; sampler, chain_to_beta, kwargs...)

swapsampler(sampler::TemperedSampler) = SwapSampler(sampler.swapstrategy)

# TODO: Do we need this now?
getsampler(samplers, I...) = getindex(samplers, I...)
getsampler(sampler::AbstractMCMC.AbstractSampler, I...) = sampler
getsampler(sampler::TemperedSampler, I...) = getsampler(sampler.sampler, I...)

chain_to_process(state::TemperedState, I...) = chain_to_process(state.swapstate, I...)
process_to_chain(state::TemperedState, I...) = process_to_chain(state.swapstate, I...)

"""
    sampler_for_chain(sampler::TemperedSampler, state::TemperedState, I...)

Return the sampler corresponding to the chain indexed by `I...`.
"""
function sampler_for_chain(sampler::TemperedSampler, state::TemperedState, I...)
    return sampler_for_process(sampler, state, chain_to_process(state, I...))
end

"""
    sampler_for_process(sampler::TemperedSampler, state::TemperedState, I...)

Return the sampler corresponding to the process indexed by `I...`.
"""
function sampler_for_process(sampler::TemperedSampler, state::TemperedState, I...)
    return _sampler_for_process_temper(sampler.sampler, state, I...)
end

# If `sampler` is a `MultiSampler`, we assume it's ordered according to chains.
_sampler_for_process_temper(sampler::MultiSampler, state, I...) = sampler.samplers[process_to_chain(state, I...)]
# Otherwise, we just use the same sampler for everything.
_sampler_for_process_temper(sampler, state, I...) = sampler

# Defer extracting the corresponding state to the `swapstate`.
state_for_process(state::TemperedState, I...) = state_for_process(state.swapstate, I...)

# Here we make the model(s) using the temperatures.
function model_for_process(sampler::TemperedSampler, model, state::TemperedState, I...)
    return make_tempered_model(sampler, model, beta_for_process(state, I...))
end

"""
    beta_for_chain(state[, I...])

Return the β corresponding to the chain indexed by `I...`.
If `I...` is not specified, the β corresponding to `β=1.0` will be returned.
"""
beta_for_chain(state::TemperedState) = beta_for_chain(state, 1)
beta_for_chain(state::TemperedState, I...) = beta_for_chain(state.chain_to_beta, I...)
# NOTE: Array impl. is useful for testing.
beta_for_chain(chain_to_beta::AbstractArray, I...) = chain_to_beta[I...] 

"""
    beta_for_process(state, I...)

Return the β corresponding to the process indexed by `I...`.
"""
beta_for_process(state::TemperedState, I...) = beta_for_process(state.chain_to_beta, state.swapstate.process_to_chain, I...)
# NOTE: Array impl. is useful for testing.
function beta_for_process(chain_to_beta::AbstractArray, proc2chain::AbstractArray, I...)
    return beta_for_chain(chain_to_beta, process_to_chain(proc2chain, I...))
end

"""
    numsteps(sampler::TemperedSampler)

Return number of inverse temperatures used by `sampler`.
"""
numtemps(sampler::TemperedSampler) = length(sampler.chain_to_beta)

# Stepping.
get_init_params(x, _) = x
get_init_params(init_params::Nothing, _) = nothing
get_init_params(init_params::AbstractVector{<:Real}, _) = copy(init_params)
get_init_params(init_params::AbstractVector{<:AbstractVector{<:Real}}, i) = init_params[i]

function transition_for_chain(transition::TemperedTransition, I...)
    chain_idx = transition.swaptransition.chain_to_process[I...]
    return transition.transition.transitions[chain_idx]
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedSampler;
    kwargs...
)
    # Create a `MultiSampler` and `MultiModel`.
    multimodel = MultiModel([
        make_tempered_model(sampler, model, sampler.chain_to_beta[i])
        for i in 1:numtemps(sampler)
    ])
    multisampler = MultiSampler([getsampler(sampler, i) for i in 1:numtemps(sampler)])
    multitransition, multistate = AbstractMCMC.step(rng, multimodel, multisampler; kwargs...)

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

    swaptransition = SwapTransition(deepcopy(swapstate.chain_to_process), deepcopy(swapstate.process_to_chain))
    return (
        TemperedTransition(swaptransition, multitransition),
        TemperedState(swapstate, multistate, sampler.chain_to_beta)
    )
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
    # We're assuming the user has given the samplers in an order according to the initial models.
    multisampler = MultiSampler(samplers_by_processes(
        ChainOrder(),
        [getsampler(sampler, i) for i in 1:numtemps(sampler)],
        state.swapstate
    ))
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

