using Distributions: StatsFuns

@concrete struct PolynomialStep
    η
    c
end
function get(step::PolynomialStep, k::Real)
    return step.c * (k + 1) ^ (-step.η)
end

struct AdaptiveState{T1<:Real,T2<:Real,P<:PolynomialStep}
    swap_target_ar::T1
    scale_unconstrained::T2
    step::P
end

"""
    weight(ρ::AdaptiveState)

Return the weight/scale to be used in the mapping `β[ℓ] ↦ β[ℓ + 1]`.

# Notes
In Eq. (13) in [^MIAS12] they use the relation

    β[ℓ + 1] = β[ℓ] * w(ρ)

with

    w(ρ) = exp(-exp(ρ))

because we want `w(ρ) ∈ (0, 1)` while `ρ ∈ ℝ`. As an alternative, we use
`StatsFuns.logistic(ρ)` which is numerically more stable than `exp(-exp(ρ))` and
leads to less extreme values, i.e. 0 or 1.

# References
[^MIAS12] Miasojedow, B., Moulines, E., & Vihola, M., Adaptive Parallel Tempering Algorithm, (2012).
"""
weight(ρ::AdaptiveState) = StatsFuns.logistic(ρ.scale_unconstrained)

function init_adaptation(
    Δ::Vector{<:Real},
    swap_target::Real,
    scale::Real,
    γ::Real
)
    Nt = length(Δ)
    step = PolynomialStep(γ, Nt - 1)
    ρs = [
        AdaptiveState(swap_target, StatsFuns.logit(scale), step)
        for _ in 1:(Nt - 1)
    ]
    return ρs
end


"""
    adapt!!(ρ::AdaptiveState, swap_ar, n)

Return increment used to update `ρ`.

Corresponds to the increment in Eq. (14) from [^MIAS12].

# References
[^MIAS12] Miasojedow, B., Moulines, E., & Vihola, M., Adaptive Parallel Tempering Algorithm, (2012).
"""
function adapt!!(ρ::AdaptiveState, swap_ar, n)
    swap_diff = swap_ar - ρ.swap_target_ar
    γ = get(ρ.step, n)
    return @set ρ.scale_unconstrained = ρ.scale_unconstrained + γ * swap_diff
end

"""
    adapt!!(ρ::AdaptiveState, Δ, k, swap_ar, n)
    adapt!!(ρ::AbstractVector{<:AdaptiveState}, Δ, k, swap_ar, n)

Return adapted state(s) given that we just proposed a swap of the `k`-th
and `(k + 1)`-th temperatures with acceptance ratio `swap_ar`.
"""
adapt!!(ρ::AdaptiveState, Δ, k, swap_ar, n) = adapt!!(ρ, swap_ar, n)
function adapt!!(ρs::AbstractVector{<:AdaptiveState}, Δ, k, swap_ar, n)
    ρs[k] = adapt!!(ρs[k], swap_ar, n)
    return ρs
end

"""
    update_inverse_temperatures(ρ::AdaptiveState, Δ_current)
    update_inverse_temperatures(ρ::AbstractVector{<:AdaptiveState}, Δ_current)

Return updated inverse temperatures computed from adaptation state(s) and `Δ_current`.

If `ρ` is a `AbstractVector`, then it should be of length `length(Δ_current) - 1`,
with `ρ[k]` corresponding to the adaptation state for the `k`-th inverse temperature.

This performs an update similar to Eq. (13) in [^MIAS12], with the only possible deviation
being how we compute the scaling factor from `ρ`: see [`weight`](@ref) for information.

# References
[^MIAS12] Miasojedow, B., Moulines, E., & Vihola, M., Adaptive Parallel Tempering Algorithm, (2012).
"""
function update_inverse_temperatures(ρ::AdaptiveState, Δ_current)
    Δ = Δ_current
    N = length(Δ)
    @assert length(ρs) ≥ N - 1 "number of adaptive states < number of temperatures"

    for ℓ in 1:N - 1
        @inbounds Δ[ℓ + 1] = Δ[ℓ] * weight(ρ)
    end
    return Δ
end

function update_inverse_temperatures(ρs::AbstractVector{<:AdaptiveState}, Δ_current)
    Δ = Δ_current
    Δ[1] = Δ_current[1]
    for ℓ in 1:length(Δ) - 1
        @inbounds Δ[ℓ + 1] = Δ[ℓ] * weight(ρs[ℓ])
    end
    return Δ
end
