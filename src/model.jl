
"""
    AbstractTemperingStrategy

Abstract type for tempering strategies.

# Implementations
- [`PowerTemperingStrategy`](@ref)
"""
abstract type AbstractTemperingStrategy end

"""
    make_tempered_model(tempering, [sampler, ]model, beta)

Return an instance representing a `model` tempered with `beta`.

The return-type depends on its usage in [`compute_logdensities`](@ref).
"""
make_tempered_model(tempering, sampler, model, beta) = make_tempered_model(tempering, model, beta)


# `PowerTemperingStrategy`.
"""
    PowerTemperingStrategy <: AbstractTemperingStrategy

A tempering strategy that raises the entire log-density to the power of `beta`.
"""
struct PowerTemperingStrategy <: AbstractTemperingStrategy end

function make_tempered_model(::PowerTemperingStrategy, model, beta)
    if !implements_logdensity(model)
        error("`make_tempered_model` is not implemented for $(typeof(model)); either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end

    return PowerTemperedLogDensityProblem(model, beta)
end
function make_tempered_model(::PowerTemperingStrategy, model::AbstractMCMC.LogDensityModel, beta)
    return AbstractMCMC.LogDensityModel(PowerTemperedLogDensityProblem(model.logdensity, beta))
end

# `PathTemperingStrategy`.
"""
    PathTemperingStrategy <: AbstractTemperingStrategy

A tempering strategy that interpolates between some reference log-density and the target log-density.
"""
struct PathTemperingStrategy <: AbstractTemperingStrategy end

function make_tempered_model(tempering::PathTemperingStrategy, model, beta)
    if !implements_logdensity(model)
        error("`make_tempered_model` is not implemented for $(typeof(model)); either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end

    return PowerTemperedLogDensityProblem(model, beta)
end
