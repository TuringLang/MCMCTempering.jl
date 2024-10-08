module MCMCTemperingAdvancedHMCExt

using MCMCTempering: MCMCTempering, Setfield
using AdvancedHMC: AdvancedHMC

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

end
