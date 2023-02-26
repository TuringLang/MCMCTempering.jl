using MCMCTempering
using Test
using Distributions
using AdvancedMH
using MCMCChains
using Bijectors
using LinearAlgebra
using AbstractMCMC
using LogDensityProblems: LogDensityProblems, logdensity, logdensity_and_gradient
using LogDensityProblemsAD
using ForwardDiff: ForwardDiff
using AdvancedMH: AdvancedMH
using AdvancedHMC: AdvancedHMC
using Turing: Turing, DynamicPPL


include("utils.jl")
include("compat.jl")
include("functions.jl")


@testset "MCMCTempering.jl" begin
    @testset "Swapping" begin
        # Chains:    process_to_chain    chain_to_process     chain_to_beta[process_to_chain[i]]
        # | | | |       1  2  3  4          1  2  3  4             1.00  0.75  0.50  0.25
        # | | | |
        #  V  | |       2  1  3  4          2  1  3  4             0.75  1.00  0.50  0.25
        #  Λ  | |
        # |  V  |       2  3  1  4          3  1  2  4             0.75  0.50  1.00  0.25
        # |  Λ  |
        # Initial values.
        process_to_chain = [1, 2, 3, 4]
        chain_to_process = [1, 2, 3, 4]
        chain_to_beta = [1.0, 0.75, 0.5, 0.25]

        # Make swap chain 1 (now on process 1) ↔ chain 2 (now on process 2)
        MCMCTempering.swap_betas!(chain_to_process, process_to_chain, 1, 2)
        # Expected result: chain 1 is now on process 2, chain 2 is now on process 1.
        target_process_to_chain = [2, 1, 3, 4]
        @test process_to_chain[chain_to_process] == 1:length(process_to_chain)
        @testset "$((process_idx, chain_idx, process_β))" for (process_idx, chain_idx, process_β) in zip(
            [1, 2, 3, 4],
            target_process_to_chain,
            chain_to_beta[target_process_to_chain]
        )
            @test MCMCTempering.process_to_chain(process_to_chain, chain_idx) == process_idx
            @test MCMCTempering.chain_to_process(chain_to_process, process_idx) == chain_idx
            @test MCMCTempering.beta_for_chain(chain_to_beta, chain_idx) == process_β
            @test MCMCTempering.beta_for_process(chain_to_beta, process_to_chain, process_idx) == process_β
        end

        # Make swap chain 2 (now on process 1) ↔ chain 3 (now on process 3)
        MCMCTempering.swap_betas!(chain_to_process, process_to_chain, 2, 3)
        # Expected result: chain 3 is now on process 1, chain 2 is now on process 3.
        target_process_to_chain = [3, 1, 2, 4]
        @test process_to_chain[chain_to_process] == 1:length(process_to_chain)
        @testset "$((process_idx, chain_idx, process_β))" for (process_idx, chain_idx, process_β) in zip(
            [1, 2, 3, 4],
            target_process_to_chain,
            chain_to_beta[target_process_to_chain]
        )
            @test MCMCTempering.process_to_chain(process_to_chain, process_idx) == chain_idx
            @test MCMCTempering.chain_to_process(chain_to_process, chain_idx) == process_idx
            @test MCMCTempering.beta_for_chain(chain_to_beta, chain_idx) == process_β
            @test MCMCTempering.beta_for_process(chain_to_beta, process_to_chain, process_idx) == process_β
        end
    end

    @testset "Simple MvNormal with no expected swaps" begin
        num_iterations = 10_000
        d = 1
        model = DistributionLogDensity(MvNormal(ones(d), I))

        # Setup non-tempered.
        sampler_rwmh = RWMH(MvNormal(zeros(d), 0.1 * I))

        # Sample.
        chain_tempered = test_and_sample_model(
            model,
            sampler_rwmh,
            [1.0, 1e-3],  # extreme temperatures -> don't exect much swapping to occur
            num_iterations=num_iterations,
            swap_every=2,
            adapt=false,
            init_params = [[0.0], [1000.0]],  # initialized far apart
            # At most 1% of swaps should be successful.
            mean_swap_rate_bound=0.01,
            compare_mean_swap_rate=≤,
        )
        # `atol` is fairly high because we haven't run this for "too" long. 
        @test mean(chain_tempered[:, 1, :]) ≈ 1 atol=0.2
    end

    @testset "GMM 1D" begin
        num_iterations = 10_000
        model = DistributionLogDensity(
            MixtureModel(Normal, [(-3, 1.5), (3, 1.5), (15, 1.5), (90, 1.5)], [0.175, 0.25, 0.275, 0.3])
        )

        # Setup non-tempered.
        sampler_rwmh = RWMH(MvNormal(0.1 * ones(1)))

        # Simple geometric ladder
        inverse_temperatures = MCMCTempering.check_inverse_temperatures(0.95 .^ (0:20))

        # There shouldn't be any swaps between two 
        chain_tempered = test_and_sample_model(
            model,
            sampler_rwmh,
            inverse_temperatures,
            num_iterations=num_iterations,
            swap_every=2,
            adapt=false,
            # At least 25% of swaps should be successful.
            mean_swap_rate_bound=0.25,
            compare_mean_swap_rate=≥,
            progress=false,
        )

        # # Compare the chains.
        # compare_chains(chain, chain_tempered, atol=1e-1, compare_std=false, compare_ess=true)
    end

    @testset "MvNormal 2D with different swap strategies" begin
        d = 2
        num_iterations = 20_000
        swap_every = 2

        μ_true = [-5.0, 5.0]
        σ_true = [1.0, √(10.0)]

        # Sampler parameters.
        inverse_temperatures = MCMCTempering.check_inverse_temperatures([0.25, 0.5, 0.75, 1.0])

        # Construct a DensityModel.
        model = DistributionLogDensity(MvNormal(μ_true, Diagonal(σ_true .^ 2)))

        # Set up our sampler with a joint multivariate Normal proposal.
        sampler = RWMH(MvNormal(zeros(d), Diagonal(σ_true .^ 2)))
        # Sample for the non-tempered model for comparison.
        chain = AbstractMCMC.sample(
            model, sampler, num_iterations;
            progress=false, chain_type=MCMCChains.Chains
        )

        # Different swap strategies to test.
        swapstrategies = [
            MCMCTempering.ReversibleSwap(),
            MCMCTempering.NonReversibleSwap(),
            MCMCTempering.SingleSwap(),
            MCMCTempering.SingleRandomSwap(),
            MCMCTempering.RandomSwap()
        ]

        @testset "$(swapstrategy)" for swapstrategy in swapstrategies
            chain_tempered = test_and_sample_model(
                model,
                sampler,
                inverse_temperatures,
                num_iterations=num_iterations,
                swap_every=swap_every,
                swapstrategy=swapstrategy,
                adapt=false,
            )
            compare_chains(chain, chain_tempered, rtol=0.1, compare_std=false, compare_ess=true)
        end
    end

    @testset "Turing.jl" begin
        # Instantiate model.
        model_dppl = DynamicPPL.TestUtils.demo_assume_dot_observe()
        # Move to unconstrained space.
        vi = DynamicPPL.link!!(DynamicPPL.VarInfo(model_dppl), model_dppl)
        # Get some initial values in unconstrained space.
        init_params = copy(vi[:])
        # Get the parameter names.
        param_names = map(Symbol, DynamicPPL.TestUtils.varnames(model_dppl))
        # Get bijector so we can get back to unconstrained space afterwards.
        b = inverse(Turing.bijector(model_dppl))
        # Construct the `LogDensityFunction` which supports LogDensityProblems.jl-interface.
        model = ADgradient(:ForwardDiff, DynamicPPL.LogDensityFunction(model_dppl, vi))

        @testset "Tempering of models" begin
            beta = 0.5
            model_tempered = MCMCTempering.make_tempered_model(model, beta)
            @test logdensity(model_tempered, init_params) ≈ beta * logdensity(model, init_params)
            @test last(logdensity_and_gradient(model_tempered, init_params)) ≈
                beta .* last(logdensity_and_gradient(model, init_params))
        end

        @testset "AdvancedHMC.jl" begin
            num_iterations = 2_000

            function create_HMCSampler(model)
                # Define a Hamiltonian system
                integrator = AdvancedHMC.Leapfrog(0.1)
                proposal = AdvancedHMC.NUTS{AdvancedHMC.MultinomialTS, AdvancedHMC.GeneralisedNoUTurn}(integrator)
                metric = AdvancedHMC.DiagEuclideanMetric(LogDensityProblems.dimension(model))
                adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.8, integrator))
                return AdvancedHMC.HMCSampler(proposal, metric, adaptor)
            end
            
            # Sample using HMC.
            chain_hmc = sample(
                model, create_HMCSampler(model), num_iterations;
                init_params=copy(init_params), progress=false,
                chain_type=MCMCChains.Chains, param_names=param_names
            )
            map_parameters!(b, chain_hmc)

            # Sample using tempered HMC.
            chain_tempered = test_and_sample_model(
                model,
                model -> create_HMCSampler(model),
                0.5 .^ collect(0:3),
                swap_strategy=MCMCTempering.ReversibleSwap(),
                num_iterations=num_iterations,
                swap_every=10,
                adapt=false,
                mean_swap_rate_bound=0,
                init_params=copy(init_params),
                param_names=param_names,
                progress=false
            )
            map_parameters!(b, chain_tempered)

            compare_chains(chain_hmc, chain_tempered, atol=0.2, compare_std=false, compare_ess=false, isbroken=false)
        end
        
        @testset "AdvancedMH.jl" begin
            num_iterations = 100_000
            d = LogDensityProblems.dimension(model)

            # Set up MALA sampler.
            σ² = 1e-2
            sampler_mh = MALA(∇ -> MvNormal(σ² * ∇, 2σ² * I))

            # Sample using MALA.
            chain_mh = AbstractMCMC.sample(
                model, sampler_mh, num_iterations;
                init_params=copy(init_params), progress=false,
                chain_type=MCMCChains.Chains, param_names=param_names
            )
            map_parameters!(b, chain_mh)

            # Sample using tempered MALA.
            chain_tempered = test_and_sample_model(
                model,
                sampler_mh,
                0.5 .^ collect(0:3),
                swap_strategy=MCMCTempering.ReversibleSwap(),
                num_iterations=num_iterations,
                swap_every=2,
                adapt=false,
                mean_swap_rate_bound=0,
                init_params=copy(init_params),
                param_names=param_names
            )
            map_parameters!(b, chain_tempered)
            
            # Need a large atol as MH is not great on its own
            compare_chains(chain_mh, chain_tempered, atol=0.4, compare_std=false, compare_ess=true, isbroken=false)
        end
    end
end
