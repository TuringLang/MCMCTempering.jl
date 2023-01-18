"""
    make_tempered_model(sampler, model, args...)

Return an instance representing a model.

The return-type depends on its usage in [`compute_tempered_logdensities`](@ref).
"""
make_tempered_model(sampler, model, beta) = make_tempered_model(model, beta)
make_tempered_model(model, beta) = TemperedLogDensityProblem(model, beta)
function make_tempered_model(model::AbstractMCMC.LogDensityModel, beta)
    return AbstractMCMC.LogDensityModel(TemperedLogDensityProblem(model.logdensity, beta))
end

