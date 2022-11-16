"""
    TemperedSampler <: AbstractMCMC.AbstractSampler

A `TemperedSampler` struct wraps an `sampler` and samples using parallel tempering.

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

Return number of temperatures used by `sampler`.
"""
numtemps(sampler::TemperedSampler) = length(sampler.inverse_temperatures)


"""
    tempered(sampler, inverse_temperatures; kwargs...)
    OR
    tempered(sampler, Nt::Integer; kwargs...)

Return tempered version of `sampler` using the provided `inverse_temperatures` or
inverse temperatures generated from `Nt` and the `swap_strategy`.

# Arguments
- `sampler` is an algorithm or sampler object to be used for underlying sampling and to apply tempering to
- The temperature schedule can be defined either explicitly or just as an integer number of temperatures, i.e. as:
  - `inverse_temperatures` containing a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
        OR
  - `Nt::Integer`, specifying the number of inverse temperatures to include in a generated `inverse_temperatures`

# Keyword arguments
- `swap_strategy::AbstractSwapStrategy` is the way in which temperature swaps are made.
- `swap_every::Integer` steps are carried out between each tempering swap step attempt

# See also
- [`TemperedSampler`](@ref)
- For more on the swap strategies:
  - [`AbstractSwapStrategy`](@ref)
  - [`StandardSwap`](@ref)
  - [`RandomPermutationSwap`](@ref)
  - [`NonReversibleSwap`](@ref)
"""
function tempered(
    sampler,
    Nt::Integer,
    swap_strategy::AbstractSwapStrategy = StandardSwap();
    kwargs...
)
    return tempered(sampler, generate_inverse_temperatures(Nt, swap_strategy); kwargs...)
end
function tempered(
    sampler,
    inverse_temperatures::Vector{<:Real},
    swap_strategy::AbstractSwapStrategy = StandardSwap();
    swap_every::Integer = 1,
    adapt::Bool = true,
    adapt_target::Real = 0.234,
    adapt_stepsize::Real = 1,
    adapt_eta::Real = 0.66,
    adapt_schedule = Geometric(),
    adapt_scale = defaultscale(adapt_schedule, inverse_temperatures),
    kwargs...
)
    inverse_temperatures = check_inverse_temperatures(inverse_temperatures)
    length(inverse_temperatures) > 1 || error("More than one inverse temperatures must be provided.")
    swap_every >= 1 || error("This must be a positive integer.")
    adaptation_states = init_adaptation(
        adapt_schedule, inverse_temperatures, adapt_target, adapt_scale, adapt_eta, adapt_stepsize
    )
    return TemperedSampler(sampler, inverse_temperatures, swap_every, swap_strategy, adapt, adaptation_states)
end
