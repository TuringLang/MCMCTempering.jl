"""
    swap_acceptance_st

Calculates and returns the swap acceptance ratio to take the chain from inverse temperature `β` to `β_proposed`
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sample` contains sampled parameters at which to calculate the log density
- `β` is the current inverse temperature of the ST algorithm
- `β_proposed` is the proposed inverse temperature to step to for further iterations
- `K` is the temperature normalising function
"""
function swap_acceptance_st(model::AbstractMCMC.AbstractModel, sample, β::Float64, β_proposed::Float64, K)
    return min(
        1, 
        (K(β_proposed) * exp(AdvancedMH.logdensity(model, sample) * β_proposed)) / (K(β) * exp(AdvancedMH.logdensity(model, sample) * β))
    )
end
# How to generalise this outside of AdvancedMH?
# TODO think of ways to calculate `K` in an intelligent way

"""
    swap_acceptance_pt

Calculates and returns the swap acceptance ratio for swapping the temperature of two chains, the `k`th and `k + 1`th
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `samplek` contains sampled parameters of the `k`th chain at which to calculate the log density
- `samplekp1` contains sampled parameters of the `k + 1`th chain at which to calculate the log density
- `βk` is the temperature of the `k`th chain
- `βkp1` is the temperature of the `k + 1`th chain PT may be swapping the `k`th chain's temperature with
"""
function swap_acceptance_pt(model::AbstractMCMC.AbstractModel, samplek, samplekp1, βk::Float64, βkp1::Float64)
    return min(
        1,
        exp(AdvancedMH.logdensity(model, samplek) * βkp1 + AdvancedMH.logdensity(model, samplekp1) * βk) / exp(AdvancedMH.logdensity(model, samplek) * βk + AdvancedMH.logdensity(model, samplekp1) * βkp1)
        # exp(abs(βk - βkp1) * abs(AdvancedMH.logdensity(model, samplek) - AdvancedMH.logdensity(model, samplekp1)))
    )
end
# How to generalise this outside of AdvancedMH?
