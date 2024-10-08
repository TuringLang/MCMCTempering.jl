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
    NoSwap


function supports_adaptation(strategy::Union{RandomSwap, SingleRandomSwap, NoSwap})
    throw(ValueError("Adaptation of the inverse temperature ladder is not currently supported under the chosen swap strategy $(strategy)."))
end
supports_adaptation(::AbstractSwapStrategy) = true

"""
    tempered(sampler, inverse_temperatures; kwargs...)
    OR
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
    num_temps::Integer;
    swap_strategy::AbstractSwapStrategy=ReversibleSwap(),
    kwargs...
)
    return tempered(
        sampler, generate_inverse_temperatures(num_temps, swap_strategy);
        swap_strategy = swap_strategy,
        kwargs...
    )
end
function tempered(
    sampler::AbstractMCMC.AbstractSampler,
    inverse_temperatures::Vector{<:Real};
    swap_strategy::AbstractSwapStrategy=ReversibleSwap(),
    steps_per_swap::Integer=1,
    adapt::Bool=false,
    adapt_target::Real=0.234,
    adapt_stepsize::Real=1,
    adapt_eta::Real=0.66,
    adapt_schedule=Geometric(),
    adapt_scale=defaultscale(adapt_schedule, inverse_temperatures),
    kwargs...
)
    adapt && supports_adaptation(swap_strategy)
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

end
