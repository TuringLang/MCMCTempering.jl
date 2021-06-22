"""
    get_scaling_val(Nt, swap_strategy)

Calculates the correct scaling factor for polynomial step size between temperatures
"""
function get_scaling_val(Nt, swap_strategy)
    # Why these?
    if swap_strategy == :standard
        scaling_val = Nt - 1
    elseif swap_strategy == :nonrev
        scaling_val = 2
    else
        scaling_val = 1
    end
    return scaling_val
end


"""
    generate_Δ(Nt, swap_strategy)

Returns a temperature ladder `Δ` containing `Nt` temperatures,
generated in accordance with the chosen `swap_strategy`
"""
function generate_Δ(Nt, swap_strategy)
    scaling_val = get_scaling_val(Nt, swap_strategy)
    Δ = zeros(Real, Nt)
    Δ[1] = 1.0
    β′ = Δ[1]
    for i ∈ 1:(Nt - 1)
        β′ += exp(scaling_val)
        Δ[i + 1] = Δ[1] / β′
    end
    return Δ
end


"""
    check_Δ(Δ)

Checks and returns a sorted `Δ` containing `{β₀, ..., βₙ}` conforming such that `1 = β₀ > β₁ > ... > βₙ ≥ 0`
"""
function check_Δ(Δ)
    if !all(zero.(Δ) .≤ Δ .≤ one.(Δ))
        error("Temperature schedule provided has values outside of the acceptable range, ensure all values are in [0, 1].")
    end
    Δ = sort(Δ; rev=true)
    if Δ[1] != one(Δ[1])
        error("Δ must contain 1, as β₀.")
    end
    return Δ
end
