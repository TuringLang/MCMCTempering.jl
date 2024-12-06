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
struct PathTemperingStrategy{D<:DistributionLogDensityProblem} <: AbstractTemperingStrategy
    "reference model"
    reference::D
end
PathTemperingStrategy(dist::Distributions.Distribution) = PathTemperingStrategy(DistributionLogDensityProblem(dist))

"""
    make_tempered_model([sampler, ]model, beta)

Return an instance representing a `model` tempered with `beta`.

The return-type depends on its usage in [`compute_logdensities`](@ref).
"""
make_tempered_model(sampler, model, beta) = make_tempered_model(model, beta)
function make_tempered_model(model, beta)
    if !implements_logdensity(model)
        error("`make_tempered_model` is not implemented for $(typeof(model)); either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end

    return TemperedLogDensityProblem(model, beta)
end
function make_tempered_model(model::AbstractMCMC.LogDensityModel, beta)
    return AbstractMCMC.LogDensityModel(TemperedLogDensityProblem(model.logdensity, beta))
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
