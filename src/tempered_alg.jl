
struct TemperedAlgorithm{A<:Turing.InferenceAlgorithm, B<:Vector{<:AbstractFloat}, C<:Integer, D<:Symbol} <: AbstractSampler
    alg           :: A
    Δ             :: B
    N_swap        :: C
    swap_strategy :: D
end

function Tempered(
    alg::InferenceAlgorithm,
    Δ::Vector{<:AbstractFloat};
    kwargs...
)
    return Tempered(alg; Δ=check_Δ(Δ), kwargs...)
end
function Tempered(
    alg::InferenceAlgorithm,
    Nt::Integer;
    swap_strategy::Symbol = :standard,
    kwargs...
)
    return Tempered(alg; Δ=generate_Δ(Nt, swap_strategy), swap_strategy=swap_strategy, kwargs...)
end
"""
    Tempered

# Arguments
- `alg` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- The temperature schedule can be defined either explicitly or just as an integer number of temperatures:
    - `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
        OR
    - `Nt` is an integer specifying the number of inverse temperatures to generate and run with

# Additional arguments
- `N_swap` steps are carried out between each tempering swap step attempt
- `swap_strategy` is the way in which temperature swaps are made, one of:
   `:standard` as in original proposed algorithm, a single randomly picked swap is proposed
   `:nonrev` alternate even/odd swaps as in Syed, Bouchard-Côté, Deligiannidis, Doucet, arXiv:1905.02939 such that a reverse swap cannot be made in immediate succession
   `:randperm` generates a permutation in order to swap in a random order
- TODO `swap_ar_target` defaults to 0.234 per REFERENCE
- TODO `store_swaps` is a flag determining whether to store the state of the chain after each swap move or not
"""
function Tempered(
    alg::InferenceAlgorithm;
    Δ::Vector{<:AbstractFloat},
    N_swap::Integer = 1,
    swap_strategy::Symbol = :standard,
    kwargs...
)
    return TemperedAlgorithm(alg, Δ, N_swap, swap_strategy)
end


function AbstractMCMC.sample(
    model::Turing.Model, # Should this be Model or AbstractModel ?
    t_alg::TemperedAlgorithm{<:Turing.InferenceAlgorithm,<:Vector{<:AbstractFloat},<:Integer,<:Symbol},
    N::Integer;
    kwargs...
)
    return AbstractMCMC.sample(Random.GLOBAL_RNG, model, tempered, N; kwargs...)
end
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Turing.Model, # Should this be Model or AbstractModel ?
    alg::TemperedAlgorithm{<:Turing.InferenceAlgorithm,<:Vector{<:AbstractFloat},<:Integer,<:Symbol},
    N::Integer;
    kwargs...
)
    return AbstractMCMC.sample(rng, model, Sampler(alg, model), N; kwargs...)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::Sampler{<:TemperedAlgorithm},
    N::Integer;
    chain_type=MCMCChains.Chains,
    resume_from=nothing,
    progress=Turing.PROGRESS[],
    kwargs...
)
    if resume_from === nothing
        return AbstractMCMC.mcmcsample(rng, model, sampler, N; chain_type=chain_type, progress=progress, kwargs...)
    else
        return resume(resume_from, N; chain_type=chain_type, progress=progress, kwargs...)
    end
end


function swap_step(state, Δ, swap_strategy)

    if swap_strategy == :standard
        k = rand(Distributions.Categorical(length(Δ) - 1)) # Pick randomly from 1, 2, ..., k-1
        A = swap_acceptance_pt(model, p_samples[k], p_samples[k + 1], Δ[Ts[k]], Δ[Ts[k + 1]])

        U = rand(Distributions.Uniform(0, 1))
        # If the proposed temperature swap is accepted according to A and U, swap the temperatures for future steps
        if U ≤ A
            temp = Ts[k]
            Ts[k] = Ts[k + 1]
            Ts[k + 1] = temp
        end
    # Do a step without sampling to record the change in temperature
    t, p_chains, p_temperatures, p_temperature_indices = parallel_step_without_sampling(model, sampler, Δ, Ts, Ntotal, p_samples, p_chains, p_temperatures, p_temperature_indices, t; kwargs...)

end

# function AbstractMCMC.step(
#     rng::AbstractRNG,

# )

#     if spl.

# end


# function AbstractMCMC.step(
#     rng::Random.AbstractRNG,
#     model::Model,
#     spl::Sampler;
#     resume_from = nothing,
#     kwargs...
# )
#     if resume_from !== nothing
#         state = loadstate(resume_from)
#         return AbstractMCMC.step(rng, model, spl, state; kwargs...)
#     end

#     # Sample initial values.
#     _spl = initialsampler(spl)
#     vi = VarInfo(rng, model, _spl)

#     # Update the parameters if provided.
#     if haskey(kwargs, :init_params)
#         initialize_parameters!(vi, kwargs[:init_params], spl)

#         # Update joint log probability.
#         # TODO: fix properly by using sampler and evaluation contexts
#         # This is a quick fix for https://github.com/TuringLang/Turing.jl/issues/1588
#         # and https://github.com/TuringLang/Turing.jl/issues/1563
#         # to avoid that existing variables are resampled
#         if _spl isa SampleFromUniform
#             model(rng, vi, SampleFromPrior())
#         else
#             model(rng, vi, _spl)
#         end
#     end

#     return initialstep(rng, model, spl, vi; kwargs...)
# end
