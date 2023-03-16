# AdvancedMH.jl
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
    return AdvancedMH.GradientTransition(params, AdvancedMH.logdensity_and_gradient(model, params)...)
end

# AdvancedHMC.jl
MCMCTempering.getparams_and_logprob(t::AdvancedHMC.Transition) = t.z.θ, t.z.ℓπ.value
MCMCTempering.getparams_and_logprob(state::AdvancedHMC.HMCState) = MCMCTempering.getparams_and_logprob(state.transition)

# TODO: Implement `state_from` instead, to avoid re-computation of gradients if possible.
function MCMCTempering.setparams_and_logprob!!(model, state::AdvancedHMC.HMCState, params, lp)
    # NOTE: Need to recompute the gradient because it might be used in the next integration step.
    hamiltonian = AdvancedHMC.Hamiltonian(state.metric, model)
    return Setfield.@set state.transition.z = AdvancedHMC.phasepoint(
        hamiltonian, params, state.transition.z.r;
        ℓκ=state.transition.z.ℓκ
    )
end

# Test this:
using AbstractMCMC: LogDensityModel

function MCMCTempering.state_from(
    model::LogDensityModel{<:MCMCTempering.TemperedLogDensityProblem},
    model_other::LogDensityModel{<:MCMCTempering.TemperedLogDensityProblem},
    state::AdvancedHMC.HMCState,
    state_other::AdvancedHMC.HMCState,
)
    beta = model.logdensity.beta
    beta_other = model_other.logdensity.beta

    z = state.transition.z
    z_other = state_other.transition.z

    params_other = z_other.θ
    logprob_other = z_other.ℓπ.value
    gradient_other = z_other.ℓπ.gradient

    # `logprob` is actually `β * actual_logprob`, and we want it to be `β_other * actual_logprob`, so:
    delta_beta = beta_other / beta
    logprob_new = delta_beta * logprob_other
    gradient_new = delta_beta .* gradient_other

    # Construct `PhasePoint`. Note that we keep `r` and `ℓκ` from the original state.
    return @set state.transition.z = AdvancedHMC.PhasePoint(
        params_other,
        z.r,
        AdvancedHMC.DualValue(logprob_new, gradient_new),
        z.ℓκ
    )
end
