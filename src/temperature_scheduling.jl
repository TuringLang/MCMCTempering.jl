
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


"""
    reconstruct_chains

Reconstructs `p_chains_fragmented` into one chain for each temperature, using `p_temperatures` to index the segments from each mixed-temp chain
- `p_chains_fragmented` is an array of chains, as sampled by the algorithm at varying temperatures
- `p_temperatures` contains the temperature history for each chain
"""
function reconstruct_chains(p_chains_frag, p_temperatures, p_temperature_indices, Δ)

    p_chains_new = p_chains_frag

    N = length(p_chains_frag[end])

    for i in 1:length(Δ)
        for j in 1:N
            p_chains_new[i][j] = p_chains_frag[p_temperature_indices[i][j]][j]
        end
    end

    return p_chains_new
end
# function reconstruct_chains(p_chains_frag, p_temperatures, p_temperature_indices, Δ)

#     p_chains_new = p_chains_frag

#     N = length(p_chains_frag)
#     p_chains = unzip(p_chains_frag)
#     p_temperatures = unzip(p_temperatures)

#     dict = Dict([Δ[i] => i for i in 1:length(Δ)])
#     make_lookup(dict) = key -> dict[key]
#     lookup = make_lookup(dict)

#     for i in 1:N
#         p_chains[i] = p_chains[i][collect(lookup.(p_temperatures[i]))]
#     end
#     temp = [[p_chains[i][j] for i=1:N] for j=1:length(Δ)]

#     return 
# end
# How to do this more efficiently?