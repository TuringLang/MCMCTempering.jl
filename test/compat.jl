# AdvancedMH.jl
MCMCTempering.getparams(transition::AdvancedMH.Transition) = transition.params
MCMCTempering.getparams(transition::AdvancedMH.GradientTransition) = transition.params

# AdvancedHMC.jl
MCMCTempering.getparams(t::AdvancedHMC.Transition) = t.z.θ
