
"""
    generate_Δ

Returns a temperature ladder `Δ` generated in accordance with the chosen `swap_strategy`

# Arguments
- `Nt` is the length of the resulting temperature ladder `Δ`
- `swap_strategy` determines the scaling of the ladder
"""
function generate_Δ(Nt, swap_strategy)

    # Why these?
    if swap_strategy == :standard
        scaling_val = Nt - 1
    elseif swap_strategy == :nonrev
        scaling_val = 2
    else
        scaling_val = 1
    end

    Δ = zeros(Float64, Nt)
    T = one(Float64) - exp(scaling_val)
    for i ∈ 1:Nt
        T += exp(scaling_val)
        Δ[i] = one(Float64) / T
    end

    return Δ

end


"""
    check_Δ

Returns a sorted `Δ` containing `{β₀, ..., βₙ}` conforming such that `0 ≤ βₙ < ... < β₁ < β₀ = 1`

# Arguments
- `Δ` contains a sequence of 'inverse temperatures' `{β₀, ..., βₙ}` to validate
"""
function check_Δ(Δ)
    if !all(zero.(Δ) .≤ Δ .≤ one.(Δ))
        error("Temperature schedule provided has values outside of the acceptable range, ensure all values are in [0, 1]")
    end
    Δ = sort(Δ; rev=true)
    if Δ[1] != one(Δ[1])
        error("Δ must contain 1, for β₀")
    end
    return Δ
end