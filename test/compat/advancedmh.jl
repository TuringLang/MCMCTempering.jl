##########################################
### Make compatible with AdvancedMH.jl ###
##########################################
# Makes the first step possible.
# This constructs the model that are passed to the respective samplers.
function MCMCTempering.make_tempered_model(sampler, m::DensityModel, β)
    return DensityModel(Base.Fix1(*, β) ∘ m.logdensity)
end

# Now we need to make swapping possible.
# This should return a callable which evaluates to the temperered logdensity.
function MCMCTempering.compute_tempered_logdensities(
    model::DensityModel,
    sampler,
    transition::AdvancedMH.Transition,
    transition_other::AdvancedMH.Transition,
    β
)
    # Just re-use computation from transition.
    # lp = transition.lp
    lp = β * AdvancedMH.logdensity(model, transition.params)
    # Compute for the other.
    lp_other = β * AdvancedMH.logdensity(model, transition_other.params)
    return lp, lp_other
end
