abstract type AbstractTemperingStrategy end

"""
    PowerTemperingStrategy

A strategy which simply raises the log-density to the power of `beta` to temper the model.
"""
struct PowerTemperingStrategy <: AbstractTemperingStrategy end

"""
    PathTemperingStrategy

A strategy which tempers the model to a reference model.

# Fields
$(FIELDS)
"""
struct PathTemperingStrategy{D} <: AbstractTemperingStrategy
    "reference model"
    reference::D
end

"""
    make_tempered_model(sampler, model, beta)
    make_tempered_model(strategy, model, beta)

Return an instance representing a `model` tempered with `beta`.

The return-type depends on its usage in [`compute_logdensities`](@ref).
"""
make_tempered_model(strategy::AbstractTemperingStrategy, model, beta) = make_tempered_model_from_strategy(strategy, model, beta)
# A wrapper for `LogDensityModel` to preserve the structure.
# HACK: Need to find a better way to do this.
function make_tempered_model(strategy::AbstractTemperingStrategy, model::AbstractMCMC.LogDensityModel, beta)
    return AbstractMCMC.LogDensityModel(make_tempered_model(strategy, model.logdensity, beta))
end

"""
    make_tempered_model_from_strategy(strategy::AbstractTemperingStrategy, model, beta)

Return an instance representing a `model` tempered with `beta` using `strategy`.

This method should be overloaded for both custom strategies and models.
"""
function make_tempered_model_from_strategy(::PowerTemperingStrategy, model, beta)
    if !implements_logdensity(model)
        error("`make_tempered_model` is not implemented for $(typeof(model)); either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end

    return TemperedLogDensityProblem(model, beta)
end
function make_tempered_model_from_strategy(strategy::PathTemperingStrategy, model, beta)
    if !implements_logdensity(model)
        error("`make_tempered_model` is not implemented for $(typeof(model)); either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end

    return PathTemperedLogDensityProblem(model, strategy.reference, beta)
end


"""
    logdensity(model, x)

Return the log-density of `model` at `x`.
"""
function logdensity(model, x)
    if !implements_logdensity(model)
        error("`logdensity` is not implemented for `$(typeof(model))`; either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end
    return LogDensityProblems.logdensity(model, x)
end
logdensity(model::AbstractMCMC.LogDensityModel, x) = LogDensityProblems.logdensity(model.logdensity, x)

"""
    sample_and_logprob(rng, model)

Return a sample and its log-density from `model`.
"""
function sample_and_logdensity(rng::Random.AbstractRNG, dist::Distributions.Distribution)
    params = rand(rng, dist)
    return (params, logpdf(dist, params))
end
