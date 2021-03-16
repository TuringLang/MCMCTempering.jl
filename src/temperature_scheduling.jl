
"""
    check_Δ

Returns a sorted `Δ` containing `{β₀, ..., βₙ}` conforming such that `0 ≤ βₙ < ... < β₁ < β₀ = 1`
- `Δ` contains a sequence of 'inverse temperatures' `{β₀, ..., βₙ}``
"""
function check_Δ(Δ)
    if !all(zero.(Δ) .≤ Δ .≤ one.(Δ))
        error("Temperature schedule provided has values outside of the acceptable range, ensure all values are in [0, 1]")
    end
    Δ = sort(Δ; rev=true)
    if Δ[1] != one(Δ[1])
        error("Δ must contain 1, for β₀")
    end
    return(Δ)
end
