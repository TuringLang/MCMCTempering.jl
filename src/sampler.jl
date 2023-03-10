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

"""
    tempered(sampler, inverse_temperatures; kwargs...)
    OR
    tempered(sampler, N_it; swap_strategy=ReversibleSwap(), kwargs...)

Return a tempered version of `sampler` using the provided `inverse_temperatures` or
inverse temperatures generated from `N_it` and the `swap_strategy`.

# Arguments
- `sampler` is an algorithm or sampler object to be used for underlying sampling and to apply tempering to
- The temperature schedule can be defined either explicitly or just as an integer number of temperatures, i.e. as:
  - `inverse_temperatures` containing a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
        OR
  - `N_it`, specifying the integer number of inverse temperatures to include in a generated `inverse_temperatures`

# Keyword arguments
- `swap_strategy::AbstractSwapStrategy` specifies the method for swapping inverse temperatures between chains
- `swap_every::Integer` steps are carried out between each attempt at a swap

# See also
- [`TemperedSampler`](@ref)
- For more on the swap strategies:
    - [`AbstractSwapStrategy`](@ref)
    - [`ReversibleSwap`](@ref)
    - [`NonReversibleSwap`](@ref)
    - [`SingleSwap`](@ref)
    - [`SingleRandomSwap`](@ref)
    - [`RandomSwap`](@ref)
    - [`NoSwap`](@ref)
"""
function tempered(
    sampler::AbstractMCMC.AbstractSampler,
    N_it::Integer;
    swap_strategy::AbstractSwapStrategy=ReversibleSwap(),
    kwargs...
)
    return tempered(sampler, generate_inverse_temperatures(N_it, swap_strategy); swap_strategy = swap_strategy, kwargs...)
end
function tempered(
    sampler::AbstractMCMC.AbstractSampler,
    inverse_temperatures::Vector{<:Real};
    swap_strategy::AbstractSwapStrategy=ReversibleSwap(),
    # TODO: Change `swap_every` to something like `number_of_iterations_per_swap`.
    steps_per_swap::Integer=1,
    adapt::Bool=false,
    adapt_target::Real=0.234,
    adapt_stepsize::Real=1,
    adapt_eta::Real=0.66,
    adapt_schedule=Geometric(),
    adapt_scale=defaultscale(adapt_schedule, inverse_temperatures),
    kwargs...
)
    !(adapt && typeof(swap_strategy) <: Union{RandomSwap, SingleRandomSwap}) || error("Adaptation of the inverse temperature ladder is not currently supported under the chosen swap strategy.")
    steps_per_swap > 0 || error("`steps_per_swap` must take a positive integer value.")
    inverse_temperatures = check_inverse_temperatures(inverse_temperatures)
    adaptation_states = init_adaptation(
        adapt_schedule, inverse_temperatures, adapt_target, adapt_scale, adapt_eta, adapt_stepsize
    )
    # NOTE: We just make a repeated sampler for `sampler_inner`.
    # TODO: Generalize. Allow passing in a `MultiSampler`, etc.
    sampler_inner = sampler^steps_per_swap
    return TemperedSampler(sampler_inner, inverse_temperatures, swap_strategy, adapt, adaptation_states)
end
