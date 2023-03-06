# AdvancedMH.jl
MCMCTempering.getparams_and_logprob(transition::AdvancedMH.Transition) = transition.params, transition.lp
function MCMCTempering.setparams_and_logprob!!(transition::AdvancedMH.Transition, params, lp)
    Setfield.@set! transition.params = params
    Setfield.@set! transition.lp = lp
    return transition
end
MCMCTempering.getparams_and_logprob(transition::AdvancedMH.GradientTransition) = transition.params, transition.lp
function MCMCTempering.setparams_and_logprob!!(transition::AdvancedMH.GradientTransition, params, lp)
    Setfield.@set! transition.params = params
    Setfield.@set! transition.lp = lp
    return transition
end

# AdvancedHMC.jl
MCMCTempering.getparams_and_logprob(t::AdvancedHMC.Transition) = t.z.θ, t.z.ℓπ.value
MCMCTempering.getparams_and_logprob(state::AdvancedHMC.HMCState) = MCMCTempering.getparams_and_logprob(state.transition)

# TODO: Implement `state_from` instead, to avoid re-computation of gradients if possible.
function MCMCTempering.setparams_and_logprob!!(state::AdvancedHMC.HMCState, params, lp)
    transition = state.transition
    Setfield.@set! transition.z.θ = params
    Setfield.@set! transition.z.ℓπ.value = lp
    return Setfield.@set state.transition = transition
end
