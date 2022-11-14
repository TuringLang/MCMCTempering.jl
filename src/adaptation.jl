using Distributions: StatsFuns

@concrete struct PolynomialStep
    η
    c
end
function get(step::PolynomialStep, k::Real)
    return step.c * (k + 1) ^ (-step.η)
end

"""
    Geometric

Specifies a geometric schedule for the inverse temperatures.

See also: [`AdaptiveState`](@ref), [`update_inverse_temperatures`](@ref), and
[`weight`](@ref).
"""
struct Geometric end

defaultscale(::Geometric, inverse_temperatures) = eltype(inverse_temperatures)(0.9)

"""
    InverselyAdditive

Specifies an additive schedule for the temperatures (not _inverse_ temperatures).

See also: [`AdaptiveState`](@ref), [`update_inverse_temperatures`](@ref), and
[`weight`](@ref).
"""
struct InverselyAdditive end

defaultscale(::InverselyAdditive, inverse_temperatures) = eltype(inverse_temperatures)(0.9)

struct AdaptiveState{S,T1<:Real,T2<:Real,P<:PolynomialStep}
    schedule_type::S
    swap_target_ar::T1
    scale_unconstrained::T2
    step::P
    n::Int
end

function AdaptiveState(swap_target_ar, scale_unconstrained, step)
    return AdaptiveState(InverselyAdditive(), swap_target_ar, scale_unconstrained, step)
end

function AdaptiveState(schedule_type, swap_target_ar, scale_unconstrained, step)
    return AdaptiveState(schedule_type, swap_target_ar, scale_unconstrained, step, 1)
end

"""
    weight(ρ::AdaptiveState{<:Geometric})

Return the weight/scale to be used in the mapping `β[ℓ] ↦ β[ℓ + 1]`.

# Notes
In Eq. (13) in [^MIAS12] they use the relation

    β[ℓ + 1] = β[ℓ] * w(ρ)

with

    w(ρ) = exp(-exp(ρ))

because we want `w(ρ) ∈ (0, 1)` while `ρ ∈ ℝ`. As an alternative, we use
`StatsFuns.logistic(ρ)` which is numerically more stable than `exp(-exp(ρ))` and
leads to less extreme values, i.e. 0 or 1.

This the same approach as mentioned in [^ATCH11].

# References
[^MIAS12]: Miasojedow, B., Moulines, E., & Vihola, M., Adaptive Parallel Tempering Algorithm, (2012).
[^ATCH11]: Atchade, Yves F, Roberts, G. O., & Rosenthal, J. S., Towards optimal scaling of metropolis-coupled markov chain monte carlo, Statistics and Computing, 21(4), 555–568 (2011).
"""
weight(ρ::AdaptiveState{<:Geometric}) = geometric_weight_constrain(ρ.scale_unconstrained)
geometric_weight_constrain(x) = StatsFuns.logistic(x)
geometric_weight_unconstrain(y) = inverse(StatsFuns.logistic)(y)

"""
    weight(ρ::AdaptiveState{<:InverselyAdditive})

Return the weight/scale to be used in the mapping `β[ℓ] ↦ β[ℓ + 1]`.
"""
weight(ρ::AdaptiveState{<:InverselyAdditive}) = inversely_additive_weight_constrain(ρ.scale_unconstrained)
inversely_additive_weight_constrain(x) = exp(-x)
inversely_additive_weight_unconstrain(y) = -log(y)

function init_adaptation(
    schedule::InverselyAdditive,
    Δ::Vector{<:Real},
    swap_target::Real,
    scale::Real,
    η::Real,
    stepsize::Real
)
    Nt = length(Δ)
    step = PolynomialStep(η, stepsize)
    # TODO: One common state or one per temperature?
    # ρs = [
    #     AdaptiveState(schedule, swap_target, inversely_additive_weight_unconstrain(scale), step)
    #     for _ in 1:(Nt - 1)
    # ]
    ρs = AdaptiveState(schedule, swap_target, log(scale), step)
    return ρs
end

function init_adaptation(
    schedule::Geometric,
    Δ::Vector{<:Real},
    swap_target::Real,
    scale::Real,
    η::Real,
    stepsize::Real
)
    Nt = length(Δ)
    step = PolynomialStep(η, stepsize)
    # TODO: One common state or one per temperature?
    # ρs = [
    #     AdaptiveState(schedule, swap_target, geometric_weight_unconstrain(scale), step)
    #     for _ in 1:(Nt - 1)
    # ]
    ρs = AdaptiveState(schedule, swap_target, geometric_weight_unconstrain(scale), step)
    return ρs
end


"""
    adapt!!(ρ::AdaptiveState, swap_ar)

Return updated `ρ` based on swap acceptance ratio `swap_ar`.

See [`update_inverse_temperatures`](@ref) to see how we compute the resulting
inverse temperatures from the adapted state `ρ`.
"""
function adapt!!(ρ::AdaptiveState, swap_ar)
    swap_diff = ρ.swap_target_ar - swap_ar
    γ = get(ρ.step, ρ.n)
    ρ_new = @set ρ.scale_unconstrained = ρ.scale_unconstrained + γ * swap_diff
    return @set ρ_new.n += 1
end

"""
    adapt!!(ρ::AdaptiveState, Δ, k, swap_ar)
    adapt!!(ρ::AbstractVector{<:AdaptiveState}, Δ, k, swap_ar)

Return adapted state(s) given that we just proposed a swap of the `k`-th
and `(k + 1)`-th temperatures with acceptance ratio `swap_ar`.
"""
adapt!!(ρ::AdaptiveState, Δ, k, swap_ar) = adapt!!(ρ, swap_ar)
function adapt!!(ρs::AbstractVector{<:AdaptiveState}, Δ, k, swap_ar)
    ρs[k] = adapt!!(ρs[k], swap_ar)
    return ρs
end

"""
    update_inverse_temperatures(ρ::AdaptiveState{<:Geometric}, Δ_current)
    update_inverse_temperatures(ρ::AbstractVector{<:AdaptiveState{<:Geometric}}, Δ_current)

Return updated inverse temperatures computed from adaptation state(s) and `Δ_current`.

This update is similar to Eq. (13) in [^MIAS12], with the only possible deviation
being how we compute the scaling factor from `ρ`: see [`weight`](@ref) for information.

If `ρ` is a `AbstractVector`, then it should be of length `length(Δ_current) - 1`,
with `ρ[k]` corresponding to the adaptation state for the `k`-th inverse temperature.

# References
[^MIAS12]: Miasojedow, B., Moulines, E., & Vihola, M., Adaptive Parallel Tempering Algorithm, (2012).
"""
function update_inverse_temperatures(ρ::AdaptiveState{<:Geometric}, Δ_current)
    Δ = similar(Δ_current)
    β₀ = Δ_current[1]
    Δ[1] = β₀

    β = inv(β₀)
    for ℓ in 1:length(Δ) - 1
        # TODO: Is it worth it to do this on log-scale instead?
        β *= weight(ρ)
        @inbounds Δ[ℓ + 1] = β
    end
    return Δ
end

function update_inverse_temperatures(ρs::AbstractVector{<:AdaptiveState{<:Geometric}}, Δ_current)
    Δ = similar(Δ_current)
    N = length(Δ)
    @assert length(ρs) ≥ N - 1 "number of adaptive states < number of temperatures"

    β₀ = Δ_current[1]
    Δ[1] = β₀

    β = β₀
    for ℓ in 1:N - 1
        # TODO: Is it worth it to do this on log-scale instead?
        β *= weight(ρs[ℓ])
        @inbounds Δ[ℓ + 1] = β
    end
    return Δ
end

"""
    update_inverse_temperatures(ρ::AdaptiveState{<:InverselyAdditive}, Δ_current)
    update_inverse_temperatures(ρ::AbstractVector{<:AdaptiveState{<:InverselyAdditive}}, Δ_current)

Return updated inverse temperatures computed from adaptation state(s) and `Δ_current`.

This update increments the temperature (not _inverse_ temperature) by a positive constant,
which is adapted by `ρ`.

If `ρ` is a `AbstractVector`, then it should be of length `length(Δ_current) - 1`,
with `ρ[k]` corresponding to the adaptation state for the `k`-th inverse temperature.
"""
function update_inverse_temperatures(ρ::AdaptiveState{<:InverselyAdditive}, Δ_current)
    Δ = similar(Δ_current)
    β₀ = Δ_current[1]
    Δ[1] = β₀

    T = inv(β₀)
    for ℓ in 1:length(Δ) - 1
        T += weight(ρ)
        @inbounds Δ[ℓ + 1] = inv(T)
    end
    return Δ
end

function update_inverse_temperatures(ρs::AbstractVector{<:AdaptiveState{<:InverselyAdditive}}, Δ_current)
    Δ = similar(Δ_current)
    N = length(Δ)
    @assert length(ρs) ≥ N - 1 "number of adaptive states < number of temperatures"

    β₀ = Δ_current[1]
    Δ[1] = β₀

    T = inv(β₀)
    for ℓ in 1:N - 1
        T += weight(ρs[ℓ])
        @inbounds Δ[ℓ + 1] = inv(T)
    end
    return Δ
end
