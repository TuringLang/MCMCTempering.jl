module MCMCTempering

import AbstractMCMC
import Distributions
import Random

using LogDensityProblems: LogDensityProblems
using ProgressLogging: ProgressLogging
using ConcreteStructs: @concrete
using Setfield: @set, @set!

using MCMCChains: MCMCChains

using InverseFunctions

using DocStringExtensions

include("logdensityproblems.jl")
include("abstractmcmc.jl")
include("model.jl")
include("adaptation.jl")
include("swapping.jl")
include("swapsampler.jl")
include("tempered_sampler.jl")
include("sampling.jl")
include("ladders.jl")
include("utils.jl")
include("bundle_samples.jl")

export tempered,
    tempered_sample,
    TemperedSampler,
    make_tempered_model,
    ReversibleSwap,
    NonReversibleSwap,
    SingleSwap,
    SingleRandomSwap,
    RandomSwap,
    NoSwap,
    PowerTemperingStrategy,
    PathTemperingStrategy,
    # External stuff.
    LogDensityProblems


# TODO: Should we make this trait-based instead?
implements_logdensity(x) = LogDensityProblems.capabilities(x) !== nothing
maybe_wrap_model(model) = implements_logdensity(model) ? AbstractMCMC.LogDensityModel(model) : model
maybe_wrap_model(model::AbstractMCMC.LogDensityModel) = model

"""
    tempered(sampler, inverse_temperatures; kwargs...)
    tempered(sampler, num_temps; swap_strategy=ReversibleSwap(), kwargs...)

Return a tempered version of `sampler` using the provided `inverse_temperatures` or
inverse temperatures generated from `num_temps` and the `swap_strategy`.

# Arguments
- `sampler` is an algorithm or sampler object to be used for underlying sampling and to apply tempering to
- The temperature schedule can be defined either explicitly or just as an integer number of temperatures, i.e. as:
  - `inverse_temperatures` containing a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
        OR
  - `num_temps`, specifying the integer number of inverse temperatures to include in a generated `inverse_temperatures`

# Keyword arguments
- `swap_strategy::AbstractSwapStrategy` specifies the method for swapping inverse temperatures between chains
- `steps_per_swap::Integer` steps are carried out between each attempt at a swap

# See also
- [`TemperedSampler`](@ref)
- For more on the swap strategies:
  - [`MCMCTempering.AbstractSwapStrategy`](@ref)
  - [`ReversibleSwap`](@ref)
  - [`NonReversibleSwap`](@ref)
  - [`SingleSwap`](@ref)
  - [`SingleRandomSwap`](@ref)
  - [`RandomSwap`](@ref)
  - [`NoSwap`](@ref)
- For more on temperig strategies:
  - [`MCMCTempering.AbstractTemperingStrategy`](@ref)
  - [`MCMCTempering.PowerTemperingStrategy`](@ref)
  - [`MCMCTempering.PathTemperingStrategy`](@ref)
"""
function tempered(
    sampler::AbstractMCMC.AbstractSampler,
    num_temps::Integer;
    swap_strategy::AbstractSwapStrategy=ReversibleSwap(),
    tempering_strategy::AbstractTemperingStrategy=PowerTemperingStrategy(),
    kwargs...
)
    return tempered(
        # TODO: add `tempering_strategy` to the `generate_inverse_temperatures` call
        sampler, generate_inverse_temperatures(num_temps, swap_strategy);
        swap_strategy = swap_strategy,
        kwargs...
    )
end
function tempered(
    sampler::AbstractMCMC.AbstractSampler,
    inverse_temperatures::Vector{<:Real};
    swap_strategy::AbstractSwapStrategy=ReversibleSwap(),
    tempering_strategy::AbstractTemperingStrategy=PowerTemperingStrategy(),
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

    # If we're working with `tempering_strategy` which is a `PathTemperingStrategy` AND `tempering_strategy.closed_form_sample`
    # then we construct the samplers explicitly, and make the reference sampler an `IIDSampler`.
    samplers = if tempering_strategy isa PathTemperingStrategy && tempering_strategy.closed_form_sample
        # Construct the samplers explicitly.
        vcat(
            fill(sampler, length(inverse_temperatures)),
            [IIDSampler(tempering_strategy.reference)]
        )
    else
        sampler_inner
    end
    # Moreover, if we're using `PathTemperingStrategy`, we also need to add an inverse temperature of 0.
    inverse_temperatures = tempering_strategy isa PathTemperingStrategy ? vcat(inverse_temperatures, 0) : inverse_temperatures

    return TemperedSampler(samplers, inverse_temperatures, swap_strategy, tempering_strategy, adapt, adaptation_states)
end

end
