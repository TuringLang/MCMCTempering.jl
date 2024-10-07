module MCMCTemperingAdvancedMHExt

using MCMCTempering: MCMCTempering, Setfield
using AdvancedMH: AdvancedMH

MCMCTempering.getparams_and_logprob(transition::AdvancedMH.Transition) = transition.params, transition.lp
function MCMCTempering.setparams_and_logprob!!(transition::AdvancedMH.Transition, params, lp)
    Setfield.@set! transition.params = params
    Setfield.@set! transition.lp = lp
    return transition
end
MCMCTempering.getparams_and_logprob(transition::AdvancedMH.GradientTransition) = transition.params, transition.lp
# TODO: Implement `state_from` instead, to avoid re-computation of gradients if possible.
function MCMCTempering.setparams_and_logprob!!(model, transition::AdvancedMH.GradientTransition, params, lp)
    # NOTE: We have to re-compute the gradient here because this will be used in the subsequent `step` for
    # the MALA sampler.
    return AdvancedMH.GradientTransition(
        params,
        AdvancedMH.logdensity_and_gradient(model, params)...,
        transition.accepted
    )
end

end
