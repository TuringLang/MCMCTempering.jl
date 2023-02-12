"""
    TemperedSampler <: AbstractMCMC.AbstractSampler

A `TemperedSampler` struct wraps a sampler upon which to apply the Parallel Tempering algorithm.

# Fields

$(FIELDS)
"""
@concrete struct TemperedSampler <: AbstractMCMC.AbstractSampler
    "sampler(s) used to target the tempered distributions"
    internal
    "collection of inverse temperatures β; β[i] correponds i-th tempered model"
    inverse_temperatures
    "number of steps of `sampler` to take before proposing swaps"
    swap_every
    "the swap strategy that will be used when proposing swaps"
    swap_strategy
    "adaptation parameters"
    adaptation_config
end

swapstrategy(sampler::TemperedSampler) = sampler.swap_strategy
get_sampler(sampler, I...) = sampler.internal[I...]

"""
    numsteps(sampler::TemperedSampler)

Return number of inverse temperatures used by `sampler`.
"""
numtemps(sampler::TemperedSampler) = length(sampler.inverse_temperatures)

"""
    tempered(sampler, inverse_temperatures; kwargs...)
    OR
    tempered(sampler, N_it; swap_strategy=StandardSwap(), kwargs...)

Return a tempered version of `sampler` using the provided `inverse_temperatures` or
inverse temperatures generated from `N_it` and the `swap_strategy`.

# Arguments
- `sampler` is an algorithm or sampler object to be used for underlying sampling and to apply tempering to
- The temperature schedule can be defined either explicitly or just as an integer number of temperatures, i.e. as:
  - `inverse_temperatures` containing a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
        OR
  - `N_it`, specifying the integer number of inverse temperatures to include in a generated `inverse_temperatures`

# Keyword arguments
- `swap_strategy::AbstractSwapStrategy` is the way in which inverse temperature swaps between chains are made
- `swap_every::Integer` steps are carried out between each attempt at a swap

# See also
- [`TemperedSampler`](@ref)
- For more on the swap strategies:
  - [`AbstractSwapStrategy`](@ref)
  - [`StandardSwap`](@ref)
  - [`NonReversibleSwap`](@ref)
  - [`RandomPermutationSwap`](@ref)
"""
function tempered(
    sampler::AbstractMCMC.AbstractSampler,
    N_it::Integer;
    swap_strategy::AbstractSwapStrategy=StandardSwap(),
    kwargs...
)
    return tempered(
        sampler,
        generate_inverse_temperatures(N_it, swap_strategy);
        swap_strategy = swap_strategy,
        kwargs...
    )
end
function tempered(
    sampler::AbstractMCMC.AbstractSampler,
    inverse_temperatures::Vector{<:Real};
    swap_strategy::AbstractSwapStrategy=StandardSwap(),
    swap_every::Integer=1,
    adapt_schedule=NoAdapt(),
    adapt_target_swap_ar::Real=0.234,
    adapt_scale=defaultscale(adapt_schedule, inverse_temperatures),
    adapt_eta::Real=0.66,
    adapt_stepsize::Real=1,
    kwargs...
)
    swap_every > 0 || error("This must be a positive integer.")
    inverse_temperatures = check_inverse_temperatures(inverse_temperatures)
    return TemperedSampler(
        [sampler for _ in inverse_temperatures],
        inverse_temperatures,
        swap_every,
        swap_strategy,
        wrap_adaptation_config(
            adapt_schedule,
            adapt_target_swap_ar,
            adapt_scale,
            adapt_eta,
            adapt_stepsize
        )
    )
end
