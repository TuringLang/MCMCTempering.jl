"""
    make_tempered_model([sampler, ]model, beta)

Return an instance representing a `model` tempered with `beta`.

The return-type depends on its usage in [`compute_tempered_logdensities`](@ref).
"""
make_tempered_model(sampler, model, beta) = make_tempered_model(model, beta)
function make_tempered_model(model, beta)
    if !implements_logdensity(model)
        error("`make_tempered_model` is not implemented for $(typeof(model)); either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end

    return TemperedModel(model, beta)
end
function make_tempered_model(model::AbstractMCMC.LogDensityModel, beta)
    return AbstractMCMC.LogDensityModel(TemperedLogDensityProblem(model.logdensity, beta))
end

