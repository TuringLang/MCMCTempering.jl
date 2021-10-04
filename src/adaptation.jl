@concrete struct PolynomialStep
    η
    c
end
function get(step::PolynomialStep, k::Real)
    step.c * (k + 1.) ^ (-step.η)
end


struct AdaptiveState{T1<:Real,T2<:Real,P<:PolynomialStep}
    swap_target_ar :: T1
    logscale       :: T2
    step           :: P
end


function init_adaptation(
    Δ::Vector{<:Real},
    swap_target::Real,
    scale::Real,
    γ::Real
)
    Nt = length(Δ)
    step = PolynomialStep(γ, Nt - 1)
    Ρ = [AdaptiveState(swap_target, log(scale), step) for _ in 1:(Nt - 1)]
    return Ρ
end


function rhos_to_ladder(Ρ, Δ)
    β′ = Δ[1]
    for i in 1:length(Ρ)
        β′ += exp(Ρ[i].logscale)
        Δ[i + 1] = Δ[1] / β′
    end
    return Δ
end


function adapt_rho(ρ::AdaptiveState, swap_ar, n)
    swap_diff = swap_ar - ρ.swap_target_ar
    γ = get(ρ.step, n)
    return γ * swap_diff
end


function adapt_ladder(P, Δ, k, swap_ar, n)
    P[k] = let Pk = P[k]
        @set Pk.logscale += adapt_rho(Pk, swap_ar, n)
    end
    return P, rhos_to_ladder(P, Δ)
end
