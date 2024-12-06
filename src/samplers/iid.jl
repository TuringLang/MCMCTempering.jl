struct IIDSampler{D} <: AbstractMCMC.AbstractSampler
    dist::D
end

struct IIDState{A,T}
    params::A
    logprob::T
end

getparams_and_logprob(s::IIDState) = (s.params, s.logprob)
setparams_and_logprob!!(state::IIDState, params, logprob) = IIDState(params, logprob)

function sample_and_logprob(rng::Random.AbstractRNG, sampler::IIDSampler{<:Distributions.Distribution})
    params = rand(rng, sampler.dist)
    return IIDState(params, logpdf(sampler.dist, params))
end

function AbstractMCMC.step(rng::Random.AbstractRNG, model::AbstractMCMC.LogDensityModel, sampler::IIDSampler, state=nothing; kwargs...)
    params, logprob = sample_and_logprob(rng, sampler)
    state = IIDState(params, logprob)
    return state, state
end
