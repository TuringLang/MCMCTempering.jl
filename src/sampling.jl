"""
    tempered_sample([rng, ], model, sampler, N, inverse_temperatures; kwargs...)
    OR
    tempered_sample([rng, ], model, sampler, N, N_it; swap_strategy=SingleSwap(), kwargs...)

Generate `N` samples from `model` using a tempered version of the provided `sampler` using the provided
`inverse_temperatures`, _or_ `N_it` inverse temperatures generated according to the `swap_strategy`.

# Arguments
- `model` is the target for sampling
- `sampler` is an algorithm or sampler object to be used for underlying sampling and to apply tempering to
- The temperature schedule can be defined either explicitly or just as an integer number of temperatures, i.e. as:
  - `inverse_temperatures` containing a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
        OR
  - `N_it`, specifying the integer number of inverse temperatures to include in a generated `inverse_temperatures`

# Keyword arguments
- `N_burnin::Integer` burn-in steps will be carried out before any swapping between chains is attempted
- `swap_strategy::AbstractSwapStrategy` is the way in which inverse temperature swaps between chains are made
- `swap_every::Integer` steps are carried out between each attempt at a swap

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
    it_arg::Union{Integer, Vector{<:Real}};
    kwargs...
)
    return tempered_sample(Random.default_rng(), model, sampler, N, it_arg; kwargs...)
end

function tempered_sample(
    rng,
    model,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer,
    N_it::Integer;
    swap_strategy::AbstractSwapStrategy = ReversibleSwap(),
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
    samples = AbstractMCMC.sample(rng, model, tempered_sampler, N; kwargs...)
    return prepare_tempered_chain(samples, model, sampler; kwargs...)
end

prepare_tempered_chain(samples, model, sampler::TemperedSampler; kwargs...) = prepare_tempered_sample(samples, model, get_sampler(sampler, 1); kwargs...)
function prepare_tempered_chain(
    samples,
    model,
    sampler::AbstractMCMC.AbstractSampler;
    chain_type=nothing,
    kwargs...
)
    samples = Vector{typeof(samples[1])}(samples[samples .!== nothing])
    if !isnothing(chain_type)
        return AbstractMCMC.bundle_samples(
            samples,
            maybe_wrap_model(model),
            sampler,
            samples[end],
            chain_type
        )
    else
        return samples
    end
end