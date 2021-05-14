function interchain_stats(chains)

    d = Dict()

    for param in chains.name_map.parameters
        μ = std(mean(chains[param], dims=1))
        σ = std(std(chains[param], dims=1))
        push!(d, param => Dict(:μ => μ, :σ => σ))
    end

    return d

end