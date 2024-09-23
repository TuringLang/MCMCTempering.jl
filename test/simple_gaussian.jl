@testset "Simple tempered Gaussian (closed form)" begin
    μ = Zeros(1)
    inverse_temperatures = MCMCTempering.check_inverse_temperatures(0.8 .^ (0:10))
    variances_true = inv.(inverse_temperatures)
    std_true_dict = map(variances_true) do v
        Dict(:param_1 => √v)
    end
    tempered_dists = [MvNormal(Zeros(1), I / β) for β in inverse_temperatures]
    tempered_multimodel = MCMCTempering.MultiModel(map(LogDensityModel ∘ DistributionLogDensity, tempered_dists))

    initial_params = zeros(length(μ))

    num_samples = 1_000
    num_burnin = num_samples ÷ 2
    thin = 10

    # Samplers.
    rwmh = RWMH(MvNormal(Zeros(1), I))
    rwmh_tempered = TemperedSampler(rwmh, inverse_temperatures)
    rwmh_product = MCMCTempering.MultiSampler(Fill(rwmh, length(tempered_dists)))
    rwmh_product_with_swap = rwmh_product ∘ MCMCTempering.SwapSampler()

    # Sample.
    @testset "TemperedSampler" begin
        chains_product = sample(
            DistributionLogDensity(tempered_dists[1]), rwmh_tempered, num_samples;
            initial_params,
            bundle_resolve_swaps=true,
            chain_type=Vector{MCMCChains.Chains},
            progress=false,
            discard_initial=num_burnin,
            thinning=thin,
        )
        test_chains_with_monotonic_variance(chains_product, Zeros(length(chains_product)), std_true_dict)
    end

    @testset "MultiSampler without swapping" begin
        chains_product = sample(
            tempered_multimodel, rwmh_product, num_samples;
            initial_params,
            chain_type=Vector{MCMCChains.Chains},
            progress=false,
            discard_initial=num_burnin,
            thinning=thin,
        )
        test_chains_with_monotonic_variance(chains_product, Zeros(length(chains_product)), std_true_dict)
    end

    @testset "MultiSampler with swapping (saveall=true)" begin
        chains_product = sample(
            tempered_multimodel, rwmh_product_with_swap, num_samples;
            initial_params,
            bundle_resolve_swaps=true,
            chain_type=Vector{MCMCChains.Chains},
            progress=false,
            discard_initial=num_burnin,
            thinning=thin,
        )
        test_chains_with_monotonic_variance(chains_product, Zeros(length(chains_product)), std_true_dict)
    end

    @testset "MultiSampler with swapping (saveall=true)" begin
        chains_product = sample(
            tempered_multimodel, Setfield.@set(rwmh_product_with_swap.saveall = Val(false)), num_samples;
            initial_params,
            chain_type=Vector{MCMCChains.Chains},
            progress=false,
            discard_initial=num_burnin,
            thinning=thin,
        )
        test_chains_with_monotonic_variance(chains_product, Zeros(length(chains_product)), std_true_dict)
    end
end

