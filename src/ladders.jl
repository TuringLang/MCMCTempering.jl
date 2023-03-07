"""
    get_scaling_val(N_it, <:AbstractSwapStrategy)

Calculates a scaling factor for polynomial step size between inverse temperatures.
"""
get_scaling_val(N_it, ::ReversibleSwap) = 2
get_scaling_val(N_it, ::NonReversibleSwap) = 2
get_scaling_val(N_it, ::SingleSwap) = N_it - 1
get_scaling_val(N_it, ::SingleRandomSwap) = N_it - 1
get_scaling_val(N_it, ::RandomSwap) = 1
get_scaling_val(N_it, ::NoSwap) = N_it - 1

"""
    generate_inverse_temperatures(N_it, swap_strategy)

Returns a temperature ladder `Δ` containing `N_it` values,
generated in accordance with the chosen `swap_strategy`.
"""
function generate_inverse_temperatures(N_it, swap_strategy)
    # Apparently, here we increase the temperature by a constant
    # factor which depends on `swap_strategy`.
    scaling_val = get_scaling_val(N_it, swap_strategy)
    Δ = Vector{Float64}(undef, N_it)
    Δ[1] = 1
    T = Δ[1]
    for i in 1:(N_it - 1)
        T += scaling_val
        Δ[i + 1] = inv(T)
    end
    return Δ
end


"""
    check_inverse_temperatures(Δ)

Checks and returns a sorted `Δ` containing `{β₀, ..., βₙ}` conforming such that `1 = β₀ > β₁ > ... > βₙ ≥ 0`
"""
function check_inverse_temperatures(Δ)
    if !all(zero.(Δ) .≤ Δ .≤ one.(Δ))
        error("The temperature ladder provided has values outside of the acceptable range, ensure all values are in [0, 1].")
    end
    Δ_sorted = sort(Δ; rev=true)
    if Δ_sorted[1] != one(Δ_sorted[1])
        error("The temperature ladder must contain 1.")
    end
    if Δ_sorted != Δ
        println("The temperature was sorted to ensure decreasing order.")
    end
    return Δ_sorted
end
