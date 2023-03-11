using MCMCDiagnosticTools, Statistics, DataFrames

"""
    to_dict(c::MCMCChains.Chains[, col::Symbol])

Return a dictionary mapping parameter names to the values in column `col` of `c`.

# Arguments
- `c`: A `MCMCChains.Chains` object.
- `col`: The column to extract values from. Defaults to the first column that is not `:parameters`.
"""
to_dict(c::MCMCChains.ChainDataFrame) = to_dict(c, first(filter(!=(:parameters), keys(c.nt))))
function to_dict(c::MCMCChains.ChainDataFrame, col::Symbol)
    df = DataFrame(c)
    return Dict(sym => df[findfirst(==(sym), df[:, :parameters]), col] for sym in df.parameters)
end

"""
    atol_for_chain(chain; significance=1e-3, kind=Statistics.mean)

Return a dictionary of absolute tolerances for each parameter in `chain`, computed
as the confidence interval width for the mean of the parameter with `significance`.
"""
function atol_for_chain(chain; significance=1e-3, kind=Statistics.mean)
    param_names = names(chain, :parameters)
    # Can reject H0 if, say, `abs(mean(chain2) - mean(chain1)) > confidence_width`.
    # Or alternatively, compare means but with `atol` set to the `confidence_width`.
    # NOTE: Failure to reject, i.e. passing the tests, does not imply that the means are equal.
    mcse = to_dict(MCMCChains.mcse(chain; kind), :mcse)
    return Dict(sym => quantile(Normal(0, mcse[sym]), 1 - significance/2) for sym in param_names)
end

thin_to(chain, n) = chain[1:length(chain) ÷ n:end]

"""
    test_means(chain, mean_true; kwargs...)

Test that the mean of each parameter in `chain` is approximately `mean_true`.

# Arguments
- `chain`: A `MCMCChains.Chains` object.
- `mean_true`: A `Real` or `AbstractDict` mapping parameter names to their true mean.
- `kwargs...`: Passed to `atol_for_chain`.
"""
function test_means(chain::MCMCChains.Chains, mean_true::Real; kwargs...)
    return test_means(chain, Dict(sym => mean_true for sym in names(chain, :parameters)); kwargs...)
end
function test_means(chain::MCMCChains.Chains, mean_true::AbstractDict; n=length(chain), kwargs...)
    chain = thin_to(chain, n)
    atol = atol_for_chain(chain; kwargs...)
    @test all(isapprox(mean(chain[sym]), 0, atol=atol[sym]) for sym in names(chain, :parameters))
end

"""
    test_std(chain, std_true; kwargs...)

Test that the standard deviation of each parameter in `chain` is approximately `std_true`.

# Arguments
- `chain`: A `MCMCChains.Chains` object.
- `std_true`: A `Real` or `AbstractDict` mapping parameter names to their true standard deviation.
- `kwargs...`: Passed to `atol_for_chain`.
"""
function test_std(chain::MCMCChains.Chains, std_true::Real; kwargs...)
    return test_std(chain, Dict(sym => std_true for sym in names(chain, :parameters)); kwargs...)
end
function test_std(chain::MCMCChains.Chains, std_true::AbstractDict; n=length(chain), kwargs...)
    chain = thin_to(chain, n)
    atol = atol_for_chain(chain; kind=Statistics.std, kwargs...)
    @info "std" (std(chain[sym]), std_true[sym], atol[sym]) for sym in names(chain, :parameters)
    @test all(isapprox(std(chain[sym]), std_true[sym], atol=atol[sym]) for sym in names(chain, :parameters))
end

"""
    test_std_monotonicity(chains; isbroken=false, kwargs...)

Test that the standard deviation of each parameter in `chains` is monotonically increasing.

# Arguments
- `chains`: A vector of `MCMCChains.Chains` objects.
- `isbroken`: If `true`, then the test will be marked as broken.
- `kwargs...`: Passed to `atol_for_chain`.
"""
function test_std_monotonicity(chains::AbstractVector{<:MCMCChains.Chains}; isbroken::Bool=false, kwargs...)
    param_names = names(first(chains), :parameters)
    # We should technically use a Bonferroni-correction here, but whatever.
    atols = [atol_for_chain(chain; kind=Statistics.std, kwargs...) for chain in chains]
    stds = [Dict(sym => std(chain[sym]) for sym in param_names) for chain in chains]

    num_chains = length(chains)
    lbs = [Dict(sym => stds[i][sym] - atols[i][sym] for sym in param_names) for i in 1:num_chains]
    ubs = [Dict(sym => stds[i][sym] + atols[i][sym] for sym in param_names) for i in 1:num_chains]

    for i = 2:num_chains
        for sym in param_names
            # If the upper-bound of the current is smaller than the lower-bound of the previous, then
            # we can reject the null hypothesis that they are orderd.
            if isbroken
                @test_broken ubs[i][sym] ≥ lbs[i - 1][sym]
            else
                @test ubs[i][sym] ≥ lbs[i - 1][sym]
            end
        end
    end
end

"""
    test_chains_with_monotonic_variance(chains, mean_true, std_true; significance=1e-3, kwargs...)

Test that the mean and standard deviation of each parameter in `chains` is approximately `mean_true`
and `std_true`, respectively. Also test that the standard deviation is monotonically increasing.

# Arguments
- `chains`: A vector of `MCMCChains.Chains` objects.
- `mean_true`: A vector of `Real` or `AbstractDict` mapping parameter names to their true mean.
- `std_true`: A vector of `Real` or `AbstractDict` mapping parameter names to their true standard deviation.
- `significance`: The significance level of the test.
- `kwargs...`: Passed to `atol_for_chain`.
"""
function test_chains_with_monotonic_variance(chains, mean_true, std_true; significance=1e-3, kwargs...)
    @testset "chain $i" for i = 1:length(chains)
        test_means(chains[i], mean_true[i]; kwargs...)
        test_std(chains[i], std_true[i]; kwargs...)
    end
    test_std_monotonicity(chains; significance=0.05)
end
