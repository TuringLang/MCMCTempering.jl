
struct PolynomialStep{T <: Real}
    η :: T
    c :: T
end
function get(step::PolynomialStep{T}, k::Real) where {T <: Real}
    step.c * (k + one(T)) ^ (-step.η)
end


struct AdaptiveState{T <: Real}
    swap_target :: T
    scale       :: Base.RefValue{T}
    step        :: PolynomialStep
end
function AdaptiveState(swap_target::T, scale::T, step::PolynomialStep) where {T <: Real}
    AdaptiveState(swap_target, Ref(log(scale)), step)
end


function init_adaptation(
    Δ::Vector{<:Real},
    swap_target::T,
    scale::T,
    γ::T
) where {T<:Real}
    Nt = length(Δ)
    step = PolynomialStep(γ, Nt - 1)
    Ρ = [AdaptiveState(swap_target, scale, step) for _ in 1:(Nt - 1)]
    return Ρ
end


function rhos_to_ladder(Ρ, Δ)
    β′ = Δ[1]
    for i in 1:length(Ρ)
        β′ += exp(Ρ[i].scale[])
        Δ[i + 1] = Δ[1] / β′
    end
    return Δ
end


function adapt_rho(ρ::AdaptiveState, swap_ar, n)
    swap_diff = swap_ar - ρ.swap_target_ar
    γ = get(ρ.step, n)
    return γ * swap_diff
end


function adapt_ladder(Ρ, Δ, Δ_state, k, swap_ar, n)
    Ρ[Δ_state[k]].scale[] += adapt_rho(Ρ[Δ_state[k]], swap_ar, n)
    return Ρ, rhos_to_ladder(Ρ, Δ)
end