
# function AbstractMCMC.sample(
#     model::AbstractPPL.AbstractProbabilisticProgram,
#     t_alg::TemperedAlgorithm,
#     N::Integer;
#     kwargs...
# )
#     return AbstractMCMC.sample(Random.GLOBAL_RNG, model, t_alg, N; kwargs...)
# end
# function AbstractMCMC.sample(
#     rng::Random.AbstractRNG,
#     model::AbstractPPL.AbstractProbabilisticProgram,
#     t_alg::TemperedAlgorithm,
#     N::Integer;
#     kwargs...
# )
#     return AbstractMCMC.sample(rng, model, DynamicPPL.Sampler(t_alg, model), N; kwargs...)
# end
# function AbstractMCMC.sample(
#     rng::Random.AbstractRNG,
#     model::AbstractPPL.AbstractProbabilisticProgram,
#     sampler::DynamicPPL.Sampler{<:TemperedAlgorithm},
#     N::Integer;
#     chain_type=MCMCChains.Chains,
#     resume_from=nothing,
#     progress=Turing.PROGRESS[],
#     kwargs...
# )
#     if resume_from === nothing
#         return AbstractMCMC.mcmcsample(rng, model, sampler, N; chain_type=chain_type, progress=progress, kwargs...)
#     else
#         return resume(resume_from, N; chain_type=chain_type, progress=progress, kwargs...)
#     end
# end


# function AbstractMCMC.sample(
#     model::AbstractPPL.AbstractProbabilisticProgram,
#     t_alg::TemperedAlgorithm,
#     parallel::AbstractMCMC.AbstractMCMCParallel,
#     N::Integer,
#     n_repeats::Integer;
#     kwargs...
# )
#     return AbstractMCMC.sample(Random.GLOBAL_RNG, model, t_alg, N; kwargs...)
# end
# function AbstractMCMC.sample(
#     rng::Random.AbstractRNG,
#     model::AbstractPPL.AbstractProbabilisticProgram,
#     t_alg::TemperedAlgorithm,
#     parallel::AbstractMCMC.AbstractMCMCParallel,
#     N::Integer,
#     n_repeats::Integer;
#     kwargs...
# )
#     return AbstractMCMC.sample(rng, model, Sampler(t_alg, model), N; kwargs...)
# end
# function AbstractMCMC.sample(
#     rng::Random.AbstractRNG,
#     model::AbstractPPL.AbstractProbabilisticProgram,
#     sampler::DynamicPPL.Sampler{<:TemperedAlgorithm},
#     parallel::AbstractMCMC.AbstractMCMCParallel,
#     N::Integer,
#     n_repeats::Integer;
#     chain_type=MCMCChains.Chains,
#     progress=Turing.PROGRESS[],
#     kwargs...
# )
#     return AbstractMCMC.mcmcsample(rng, model, sampler, parallel, N, n_repeats; chain_type=chain_type, progress=progress, kwargs...)
# end


# function AbstractMCMC.bundle_samples(
#     ts::Vector,
#     model::AbstractMCMC.AbstractModel,
#     spl::DynamicPPL.Sampler{<:TemperedAlgorithm},
#     state,
#     chain_type::Union{Type{MCMCChains.Chains},Type{Vector{NamedTuple}}};
#     kwargs...
# )
#     return AbstractMCMC.bundle_samples(ts, model, DynamicPPL.Sampler(spl.alg.alg, model), state, chain_type)
# end
