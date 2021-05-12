
function AbstractMCMC.sample(
    model::DynamicPPL.Model,
    t_alg::TemperedAlgorithm,
    N::Integer;
    kwargs...
)
    return AbstractMCMC.sample(Random.GLOBAL_RNG, model, t_alg, N; kwargs...)
end
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    t_alg::TemperedAlgorithm,
    N::Integer;
    chain_type=MCMCChains.Chains,
    kwargs...
)
    return AbstractMCMC.sample(rng, model, DynamicPPL.Sampler(t_alg, model), N; chain_type=chain_type, kwargs...)
end


function AbstractMCMC.sample(
    model::DynamicPPL.Model,
    t_alg::TemperedAlgorithm,
    parallel::AbstractMCMC.AbstractMCMCParallel,
    N::Integer,
    n_repeats::Integer;
    kwargs...
)
    return AbstractMCMC.sample(Random.GLOBAL_RNG, model, t_alg, N; kwargs...)
end
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    t_alg::TemperedAlgorithm,
    parallel::AbstractMCMC.AbstractMCMCParallel,
    N::Integer,
    n_repeats::Integer;
    chain_type=MCMCChains.Chains,
    kwargs...
)
    return AbstractMCMC.sample(rng, model, Sampler(t_alg, model), N; chain_type=chain_type, kwargs...)
end


function AbstractMCMC.bundle_samples(
    ts::Vector,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:TemperedAlgorithm},
    state::TemperedState,
    chain_type::Union{Type{MCMCChains.Chains},Type{Vector{NamedTuple}}};
    kwargs...
)
    return AbstractMCMC.bundle_samples(ts, model, DynamicPPL.Sampler(spl.alg.alg, model), state, chain_type)
end
