
"""
    generate_Δ

Returns a temperature schedule `Δ` generated in accordance with the chosen `swap_strategy` and `swap_ar_target`

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
    return Δ
end


"""
    setup_models

Returns an array of `length(Δ)` models with a tempered density for each `β` in `Δ`
"""
# function setup_models(model, Δ, sampler::DynamicPPL.Sampler{<:DynamicPPL.InferenceAlgorithm})
#     models = Vector{typeof(model), length(Δ)}
#     for i in 1:length(Δ)
        
#     end
# end
# function setup_models(model, Δ, sampler::AbstractMCMC.AbstractSampler)
#     models = Vector{typeof(model), length(Δ)}
#     for i in 1:length(Δ)
#         f(θ) = model.logdensity(θ) * Δ[i]
#         models[i] = AdvancedMH.DensityModel(f)
#     end
#     return models
# end


"""
    get_tempered_densities

Returns an array of `length(Δ)` models with a tempered density for each `β` in `Δ`
"""
# function get_tempered_densities(model, Δ, sampler::DynamicPPL.Sampler{<:DynamicPPL.InferenceAlgorithm})
#     tempered_densities = Vector{undef, length(Δ)}
#     for i in 1:length(Δ)
#         ctx = DynamicPPL.MiniBatchContext(
#             DynamicPPL.DefaultContext(),
#             Δ[i]
#         )
#         varinfo_init = DynamicPPL.VarInfo(model, ctx)
#         function logπ(z)
#             varinfo = DynamicPPL.VarInfo(varinfo_init, DynamicPPL.SampleFromUniform(), z)
#             model(varinfo)

#             return DynamicPPL.getlogp(varinfo)
#         end
#         tempered_densities[i] = logπ
#     end
#     tempered_densities
# end

function get_tempered_densities(model, Δ, sampler::AbstractMCMC.AbstractSampler)
    tempered_densities = Vector{undef, length(Δ)}
    for i in 1:length(Δ)
        f(θ) = model.logdensity(θ) * Δ[i]
        tempered_densities[i] = f
    end
    return tempered_densities
end


"""
    reconstruct_chains

Reconstructs `p_chains_fragmented` into one chain for each temperature, using `p_temperatures` to index the segments from each mixed-temp chain
- `p_chains_fragmented` is an array of chains, as sampled by the algorithm at varying temperatures
- `p_temperatures` contains the temperature history for each chain
"""
function reconstruct_chains(p_chains_frag, p_temperature_indices, Δ)

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