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
    init_internal_sampler(
        proto_sampler::Union{<:Function,Vector{<:AbstractMCMC.AbstractSampler},AbstractMCMC.AbstractSampler},
        inverse_temperatures::Union{Integer, Vector{<:Real}};
        model=nothing,
        shared_sampler::Bool=false
    )

Returns an `internal_sampler` depending on the nature of `proto_sampler`, which can take one of the following forms:
  - a single `sampler::AbstractMCMC`
  - a list of `n` (== number of inverse temperatures) `samplers::Vector{AbstractMCMC.AbstractSampler}`
  - a partial `Function` of `model` capable of returning an `AbstractSampler`, i.e. `model -> create_sampler(model, ...)`,
  note that you must provide the `model` kwarg to the call of `tempered` / `tempered_sample` when using this option
"""
function init_internal_sampler(
    sampler_template::Function,
    inverse_temperatures::Union{Integer, Vector{<:Real}};
    model=nothing,
    kwargs...
)
    !isnothing(model) || error("You must provide the `model` kwarg to `tempered` when providing a template function for sampler initialisation.")
    return [sampler_template(MCMCTempering.make_tempered_model(model, β)) for β in inverse_temperatures]
end

function init_internal_sampler(
    sampler::AbstractMCMC.AbstractSampler,
    inverse_temperatures::Union{Integer, Vector{<:Real}};
    shared_sampler::Bool=false,
    kwargs...
)
    return shared_sampler ? sampler : [deepcopy(sampler) for _ in eachindex(inverse_temperatures)]
end

function init_internal_sampler(
    samplers::Vector{<:AbstractMCMC.AbstractSampler},
    inverse_temperatures::Union{Integer, Vector{<:Real}};
    kwargs...
)
    length(samplers) == length(inverse_temperatures) || error("When providing a list of `samplers`, you must ensure the number provided is equal to the number of `inverse_temperatures`.")
    return samplers
end

"""
    tempered(proto_sampler, inverse_temperatures; kwargs...)
    OR
    tempered(proto_sampler, N_it; swap_strategy=ReversibleSwap(), kwargs...)

Returns a tempered sampler containing the realisation of `proto_sampler` using the provided
`inverse_temperatures` or inverse temperatures generated from `N_it` and the `swap_strategy`.

# Arguments
- `proto_sampler` is used for underlying exploration within each tempered chain, it can take the form of:
  - a single `sampler::AbstractMCMC`
  - a list of `n` (== number of inverse temperatures) `samplers::Vector{AbstractMCMC.AbstractSampler}`
  - a partial `Function` of `model` capable of returning an `AbstractSampler`, i.e. `model -> create_sampler(model, ...)`,
    note that you must provide the `model` kwarg to the call of `tempered` / `tempered_sample` when using this option
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
    proto_sampler::Union{<:Function,Vector{<:AbstractMCMC.AbstractSampler},AbstractMCMC.AbstractSampler},
    N_it::Integer;
    swap_strategy::AbstractSwapStrategy=ReversibleSwap(),
    kwargs...
)
    return tempered(proto_sampler, generate_inverse_temperatures(N_it, swap_strategy); swap_strategy = swap_strategy, kwargs...)
end
function tempered(
    proto_sampler::Union{<:Function,Vector{<:AbstractMCMC.AbstractSampler},AbstractMCMC.AbstractSampler},
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
    !(adapt && typeof(swap_strategy) <: Union{RandomSwap, SingleRandomSwap}) || error("Adaptation of the inverse temperature ladder is not currently supported under the chosen swap strategy.")
    swap_every > 1 || error("`swap_every` must take a positive integer value greater than 1.")
    inverse_temperatures = check_inverse_temperatures(inverse_temperatures)
    adaptation_states = init_adaptation(
        adapt_schedule, inverse_temperatures, adapt_target, adapt_scale, adapt_eta, adapt_stepsize
    )
    internal_sampler = init_internal_sampler(proto_sampler, inverse_temperatures; kwargs...)
    return TemperedSampler(internal_sampler, inverse_temperatures, swap_every, swap_strategy, adapt, adaptation_states)
end
