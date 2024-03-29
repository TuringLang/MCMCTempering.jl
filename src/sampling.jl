"""
    tempered_sample([rng, ], model, sampler, N, inverse_temperatures; kwargs...)
    OR
    tempered_sample([rng, ], model, sampler, N, N_it; swap_strategy=SingleSwap(), kwargs...)

Generate `N` samples from `model` using a tempered version of the provided `sampler`.
Provide either `inverse_temperatures` or `N_it` (and a `swap_strategy`) to generate some

# Keyword arguments
- `N_burnin::Integer` burn-in steps will be carried out before any swapping between chains is attempted
- `swap_strategy::AbstractSwapStrategy` specifies the method for swapping inverse temperatures between chains
- `steps_per_swap::Integer` steps are carried out between each attempt at a swap

# See also
- [`tempered`](@ref)
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
function tempered_sample(
    model,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer,
    arg::Union{Integer, Vector{<:Real}};
    kwargs...
)
    return tempered_sample(Random.default_rng(), model, sampler, N, arg; kwargs...)
end

function tempered_sample(
    rng,
    model,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer,
    N_it::Integer;
    swap_strategy::AbstractSwapStrategy = SingleSwap(),
    kwargs...
)
    return tempered_sample(model, sampler, N, generate_inverse_temperatures(N_it, swap_strategy); swap_strategy=swap_strategy, kwargs...)
end

function tempered_sample(
    rng,
    model,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer,
    inverse_temperatures::Vector{<:Real};
    kwargs...
)
    tempered_sampler = tempered(sampler, inverse_temperatures; kwargs...)
    return AbstractMCMC.sample(rng, model, tempered_sampler, N; kwargs...)
end
