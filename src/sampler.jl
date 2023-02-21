"""
    TemperedSampler <: AbstractMCMC.AbstractSampler

A `TemperedSampler` struct wraps a sampler upon which to apply the Parallel Tempering algorithm.

# Fields

$(FIELDS)
"""
@concrete struct TemperedSampler <: AbstractMCMC.AbstractSampler
    "sampler(s) used to target the tempered distributions"
    sampler
    "collection of inverse temperatures β; β[i] correponds i-th tempered model"
    inverse_temperatures
    "number of steps of `sampler` to take before proposing swaps"
    swap_every
    "the swap strategy that will be used when proposing swaps"
    swap_strategy
    # TODO: This should be replaced with `P` just being some `NoAdapt` type.
    "boolean flag specifying whether or not to adapt"
    adapt
    "adaptation parameters"
    adaptation_states
end

swapstrategy(sampler::TemperedSampler) = sampler.swap_strategy

getsampler(samplers, I...) = getindex(samplers, I...)
getsampler(sampler::AbstractMCMC.AbstractSampler, I...) = sampler
getsampler(sampler::TemperedSampler, I...) = getsampler(sampler.sampler, I...)

"""
    numsteps(sampler::TemperedSampler)

Return number of inverse temperatures used by `sampler`.
"""
numtemps(sampler::TemperedSampler) = length(sampler.inverse_temperatures)

"""
    sampler_for_chain(sampler::TemperedSampler, state::TemperedState[, I...])

Return the sampler corresponding to the chain indexed by `I...`.
If `I...` is not specified, the sampler corresponding to `β=1.0` will be returned.
"""
sampler_for_chain(sampler::TemperedSampler, state::TemperedState) = sampler_for_chain(sampler, state, 1)
function sampler_for_chain(sampler::TemperedSampler, state::TemperedState, I...)
    return getsampler(sampler.sampler, chain_to_process(state, I...))
end

"""
    sampler_for_process(sampler::TemperedSampler, state::TemperedState, I...)

Return the sampler corresponding to the process indexed by `I...`.
"""
function sampler_for_process(sampler::TemperedSampler, state::TemperedState, I...)
    return getsampler(sampler.sampler, I...)
end

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
    swap_every::Integer=10,
    adapt::Bool=false,
    adapt_target::Real=0.234,
    adapt_stepsize::Real=1,
    adapt_eta::Real=0.66,
    adapt_schedule=Geometric(),
    adapt_scale=defaultscale(adapt_schedule, inverse_temperatures),
    kwargs...
)
    swap_every > 1 || error("`swap_every` must take a positive integer value greater than 1.")
    inverse_temperatures = check_inverse_temperatures(inverse_temperatures)
    adaptation_states = init_adaptation(
        adapt_schedule, inverse_temperatures, adapt_target, adapt_scale, adapt_eta, adapt_stepsize
    )
    return TemperedSampler(sampler, inverse_temperatures, swap_every, swap_strategy, adapt, adaptation_states)
end
