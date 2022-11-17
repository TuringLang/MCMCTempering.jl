##########################################
### Make compatible with AdvancedMH.jl ###
##########################################
# Makes the first step possible.
# This constructs the model that are passed to the respective samplers.
function MCMCTempering.make_tempered_model(sampler, m::DensityModel, β)
    return DensityModel(Base.Fix1(*, β) ∘ m.logdensity)
end

# Now we need to make swapping possible, which requires computing
# the log density of the tempered model at the candidate states.
function MCMCTempering.compute_tempered_logdensities(
    model::DensityModel,
    sampler,
    transition::AdvancedMH.Transition,
    transition_other::AdvancedMH.Transition,
    β
)
    lp = β * AdvancedMH.logdensity(model, transition.params)
    lp_other = β * AdvancedMH.logdensity(model, transition_other.params)
    return lp, lp_other
end
