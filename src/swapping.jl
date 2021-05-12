
function swap_βs(Δ_state, k)
    temp = Δ_state[k]
    Δ_state[k] = Δ_state[k + 1]
    Δ_state[k + 1] = temp
    return Δ_state
end


# """
#     swap_acceptance_st

# Calculates and returns the swap acceptance ratio to take the chain from inverse temperature `β` to `β_proposed`
# - `model` an AbstractModel implementation defining the density likelihood for sampling
# - `sample` contains sampled parameters at which to calculate the log density
# - `β` is the current inverse temperature of the ST algorithm
# - `β_proposed` is the proposed inverse temperature to step to for further iterations
# - `K` is the temperature normalising function
# """
# function swap_acceptance_st(model::AbstractMCMC.AbstractModel, sample, β::Float64, β_proposed::Float64, K)
#     return min(
#         1, 
#         (K(β_proposed) * exp(AdvancedMH.logdensity(model, sample) * β_proposed)) / (K(β) * exp(AdvancedMH.logdensity(model, sample) * β))
#     )
# end
# # TODO think of ways to calculate `K` in an intelligent way

"""
    swap_acceptance_pt

Calculates and returns the swap acceptance ratio for swapping the temperature of two chains, the `k`th and `k + 1`th
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `samplek` contains sampled parameters of the `k`th chain at which to calculate the log density
- `samplekp1` contains sampled parameters of the `k + 1`th chain at which to calculate the log density
- `θk` is the temperature of the `k`th chain
- `θkp1` is the temperature of the `k + 1`th chain PT may be swapping the `k`th chain's temperature with
"""
function swap_acceptance_pt(logπk, logπkp1, θk, θkp1)
    return min(
        1,
        exp(logπkp1(θk) + logπk(θkp1)) / exp(logπk(θk) + logπkp1(θkp1))
        # exp(abs(βk - βkp1) * abs(AdvancedMH.logdensity(model, samplek) - AdvancedMH.logdensity(model, samplekp1)))
    )
end


function swap_attempt(model, sampler, states, k, Δ, Δ_state)

    logπk = make_tempered_logπ(DynamicPPL.Model(model.name, TemperedEval(model, Δ[Δ_state[k]]), model.args, model.defaults), sampler, get_vi(states, k))
    logπkp1 = make_tempered_logπ(DynamicPPL.Model(model.name, TemperedEval(model, Δ[Δ_state[k + 1]]), model.args, model.defaults), sampler, get_vi(states, k + 1))

    θk = get_θ(states, k, sampler)
    θkp1 = get_θ(states, k + 1, sampler)

    @show θk, θkp1

    A = swap_acceptance_pt(logπk, logπkp1, θk, θkp1)
    U = rand(Distributions.Uniform(0, 1))
    # If the proposed temperature swap is accepted according to A and U, swap the temperatures for future steps
    if U ≤ A
        Δ_state = swap_βs(Δ_state, k)
    end
    return Δ_state

end