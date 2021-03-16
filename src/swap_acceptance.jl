"""
    swap_acceptance_st

Calculates the swap acceptance ratio for two temperatures `β` and `β_proposed`
"""
function swap_acceptance_st(model, params, β, β_proposed, K)
    return min(
        1, 
        (K(β_proposed) * exp(AdvancedMH.logdensity(model, params) * β_proposed)) / (K(β) * exp(AdvancedMH.logdensity(model, params) * β))
    )
end

function swap_acceptance_pt(model, paramsk, paramskp1, βk, βkp1)
    return min(
        1,
        exp(AdvancedMH.logdensity(model, paramsk) * βkp1 + AdvancedMH.logdensity(model, paramskp1) * βk) / exp(AdvancedMH.logdensity(model, paramsk) * βk + AdvancedMH.logdensity(model, paramskp1) * βkp1)
    )
end