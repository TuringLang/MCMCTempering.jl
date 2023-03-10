@testset "Simple tempered Gaussian (closed form)" begin
    μ = Zeros(1)
    inverse_temperatures = MCMCTempering.check_inverse_temperatures(0.8 .^ (0:10))
    tempered_dists = [MvNormal(Zeros(1), I / β) for β in inverse_temperatures]
    tempered_multimodel = MCMCTempering.MultiModel(map(LogDensityModel ∘ DistributionLogDensity, tempered_dists))

    init_params = zeros(length(μ))

    function test_chains(chains)
        means = map(mean ∘ Array, chains)
        variances = map(var ∘ Array, chains)
        # `variances` should be monotonically increasing
        # TODO: Be clever with these thresholds. Probably good idea: scale tolerances wrt. variances of target.
        @test all(diff(variances) .> 0)
        @test all(isapprox.(means, 0, atol=0.5))
        @test isapprox(variances, inv.(inverse_temperatures), rtol=0.15)
    end

    # Samplers.
    rwmh = RWMH(MvNormal(Ones(1)))
    rwmh_tempered = TemperedSampler(rwmh, inverse_temperatures)
    rwmh_product = MCMCTempering.MultiSampler(Fill(rwmh, length(tempered_dists)))
    rwmh_product_with_swap = rwmh_product ∘ MCMCTempering.SwapSampler()

    # Sample.
    @testset "TemperedSampler" begin
        chains_tempered = sample(
            DistributionLogDensity(tempered_dists[1]), rwmh_tempered, 10_000;
            init_params,
            bundle_resolve_swaps=true,
            chain_type=Vector{MCMCChains.Chains},
            progress=false
        )
        test_chains(chains_tempered)
    end

    @testset "MultiSampler without swapping" begin
        chains_product = sample(
            tempered_multimodel, rwmh_product, 10_000;
            init_params,
            chain_type=Vector{MCMCChains.Chains},
            progress=false
        )
        test_chains(chains_product)
    end

    @testset "MultiSampler with swapping (saveall=true)" begin
        chains_product = sample(
            tempered_multimodel, rwmh_product_with_swap, 10_000;
            init_params,
            bundle_resolve_swaps=true,
            chain_type=Vector{MCMCChains.Chains},
            progress=false
        )
        test_chains(chains_product)
    end

    @testset "MultiSampler with swapping (saveall=true)" begin
        chains_product = sample(
            tempered_multimodel, Setfield.@set(rwmh_product_with_swap.saveall = Val(false)), 10_000;
            init_params,
            chain_type=Vector{MCMCChains.Chains},
            progress=false
        )
        test_chains(chains_product)
    end
end
