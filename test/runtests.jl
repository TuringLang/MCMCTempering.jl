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


"""
    test_and_sample_model(model, sampler, inverse_temperatures[, swap_strategy]; kwargs...)

Run the tempered version of `sampler` on `model` and return the resulting chain.

Several properties of the tempered sampler are tested before returning:
- No invalid swappings has occured.
- Swaps were successfully performed at least a given portion of the time.

# Arguments
- `model`: The model to temper and sample from.
- `sampler`: The sampler to temper and use to sample from `model`.
- `inverse_temperatures`: The inverse temperatures to for tempering..
- `swap_strategy`: The swap strategy to use.

# Keyword arguments
- `num_iterations`: The number of iterations to run the sampler for. Defaults to `2_000`.
- `swap_every`: The number of iterations between each swap attempt. Defaults to `2`.
- `adapt_target`: The target acceptance rate for the swaps. Defaults to `0.234`.
- `adapt_rtol`: The relative tolerance for the check of average swap acceptance rate and target swap acceptance rate. Defaults to `0.1`.
- `adapt_atol`: The absolute tolerance for the check of average swap acceptance rate and target swap acceptance rate. Defaults to `0.05`.
- `mean_swap_rate_bound`: A bound on the acceptance rate of swaps performed, e.g. if set to `0.1` and `compare_mean_swap_rate=≥` then at least 10% of attempted swaps should be accepted. Defaults to `0.1`.
- `compare_mean_swap_rate`: a binary function for comparing average swap rate against `mean_swap_rate_bound`. Defaults to `≥`.
- `init_params`: The initial parameters to use for the sampler. Defaults to `nothing`.
- `param_names`: The names of the parameters in the chain; used to construct the resulting chain. Defaults to `missing`.
- `progress`: Whether to show a progress bar. Defaults to `false`.
- `kwargs...`: Additional keyword arguments to pass to `MCMCTempering.tempered`.
"""
function test_and_sample_model(
    model,
    sampler,
    inverse_temperatures,
    swap_strategy=MCMCTempering.SingleSwap();
    mean_swap_rate_bound=0.1,
    compare_mean_swap_rate=≥,
    num_iterations=2_000,
    swap_every=2,
    adapt_target=0.234,
    adapt_rtol=0.1,
    adapt_atol=0.05,
    init_params=nothing,
    param_names=missing,
    progress=false,
    kwargs...
)
    # TODO: Remove this when no longer necessary.
    num_iterations_tempered = Int(ceil(num_iterations * swap_every ÷ (swap_every - 1)))

    # Make the tempered sampler.
    sampler_tempered = tempered(
        sampler,
        inverse_temperatures;
        swap_strategy=swap_strategy,
        swap_every=swap_every,
        adapt_target=adapt_target,
        kwargs...
    )

    # Store the states.
    states_tempered = []
    callback = StateHistoryCallback(states_tempered)

    # Sample.
    samples_tempered = AbstractMCMC.sample(
        model, sampler_tempered, num_iterations_tempered;
        callback=callback, progress=progress, init_params=init_params
    )

    # Extract the states that were swapped.
    states_swapped = filter(Base.Fix2(getproperty, :is_swap), states_tempered)
    # Swap acceptance ratios should be compared against the target acceptance in case of adaptation.
    swap_acceptance_ratios = mapreduce(
        collect ∘ values ∘ Base.Fix2(getproperty, :swap_acceptance_ratios),
        vcat,
        states_swapped
    )
    # Check that adaptation did something useful.
    if sampler_tempered.adapt
        swap_acceptance_ratios = map(Base.Fix1(min, 1.0) ∘ exp, swap_acceptance_ratios)
        empirical_acceptance_rate = sum(swap_acceptance_ratios) / length(swap_acceptance_ratios)
        @test adapt_target ≈ empirical_acceptance_rate atol = adapt_atol rtol = adapt_rtol

        # TODO: Maybe check something related to the temperatures themselves in case of adaptation.
        # E.g. converged values shouldn't all be 0 or something.
        # βs = mapreduce(Base.Fix2(getproperty, :inverse_temperatures), hcat, states)
    end

    # Extract the history of chain indices.
    process_to_chain_history_list = map(states_tempered) do state
        state.process_to_chain
    end
    process_to_chain_history = permutedims(reduce(hcat, process_to_chain_history_list), (2, 1))

    # Check that the swapping has been done correctly.
    process_to_chain_uniqueness = map(states_tempered) do state
        length(unique(state.process_to_chain)) == length(state.process_to_chain)
    end
    @test all(process_to_chain_uniqueness)

    # For the currently implemented strategies, the index process should not move by more than 1.
    @test all(abs.(diff(process_to_chain_history[:, 1])) .≤ 1)

    chain_to_process_uniqueness = map(states_tempered) do state
        length(unique(state.chain_to_process)) == length(state.chain_to_process)
    end
    @test all(chain_to_process_uniqueness)

    # Tests that we have at least swapped some times (say at least 10% of attempted swaps).
    swap_success_indicators = map(eachrow(diff(process_to_chain_history; dims=1))) do row
        # Some of the strategies performs multiple swaps in a swap-iteration,
        # but we want to count the number of iterations for which we had a successful swap,
        # i.e. only count non-zero elements in a row _once_. Hence the `min`.
        min(1, sum(abs, row))
    end
    @test compare_mean_swap_rate(
        sum(swap_success_indicators),
        (num_iterations_tempered / swap_every) * mean_swap_rate_bound
    )

    # Compare the tempered sampler to the untempered sampler.
    state_tempered = states_tempered[end]
    chain_tempered = AbstractMCMC.bundle_samples(
        samples_tempered,
        MCMCTempering.maybe_wrap_model(model),
        sampler_tempered.sampler,
        MCMCTempering.state_for_chain(state_tempered),
        MCMCChains.Chains;
        param_names=param_names
    )

    return chain_tempered
end

function compare_chains(
    chain::MCMCChains.Chains, chain_tempered::MCMCChains.Chains;
    atol=1e-6, rtol=1e-6,
    compare_std=true,
    compare_ess=true,
    isbroken=false
)
    desc = describe(chain)[1].nt
    desc_tempered = describe(chain_tempered)[1].nt

    # Compare the means.
    if isbroken
        @test_broken desc.mean ≈ desc_tempered.mean atol = atol rtol = rtol
    else
        @test desc.mean ≈ desc_tempered.mean atol = atol rtol = rtol
    end

    # Compare the std. of the chains.
    if compare_std
        if isbroken
            @test_broken desc.std ≈ desc_tempered.std atol = atol rtol = rtol
        else
            @test desc.std ≈ desc_tempered.std atol = atol rtol = rtol
        end
    end

    # Compare the ESS.
    if compare_ess
        ess = MCMCChains.ess_rhat(chain).nt.ess
        ess_tempered = MCMCChains.ess_rhat(chain_tempered).nt.ess
        # HACK: Just make sure it's not doing _horrible_. Though we'd hope it would
        # actually do better than the internal sampler.
        if isbroken
            @test_broken all(ess .≥ ess_tempered .* 0.5)
        else
            @test all(ess .≥ ess_tempered .* 0.5)
        end
    end
end

@testset "MCMCTempering.jl" begin
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
        samples = AbstractMCMC.sample(model, sampler, num_iterations; progress=false)
        chain = AbstractMCMC.bundle_samples(
            samples, MCMCTempering.maybe_wrap_model(model), sampler, samples[1], MCMCChains.Chains
        )

        # Different swap strategies to test.
        swapstrategies = [
            MCMCTempering.SingleSwap(),
            MCMCTempering.RandomSwap(),
            MCMCTempering.NonReversibleSwap()
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
        b = inv(Turing.bijector(model_dppl))
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
            num_iterations = 1_000

            # Set up HMC smpler.
            initial_ϵ = 0.1
            integrator = AdvancedHMC.Leapfrog(initial_ϵ)
            proposal = AdvancedHMC.NUTS{AdvancedHMC.MultinomialTS, AdvancedHMC.GeneralisedNoUTurn}(integrator)
            metric = AdvancedHMC.DiagEuclideanMetric(LogDensityProblems.dimension(model))
            sampler_hmc = AdvancedHMC.HMCSampler(proposal, metric)

            # Sample using HMC.
            samples_hmc = sample(model, sampler_hmc, num_iterations; init_params=copy(init_params), progress=false)
            chain_hmc = AbstractMCMC.bundle_samples(
                samples_hmc, MCMCTempering.maybe_wrap_model(model), sampler_hmc, samples_hmc[1], MCMCChains.Chains;
                param_names=param_names,
            )
            map_parameters!(b, chain_hmc)

            # Sample using tempered HMC.
            chain_tempered = test_and_sample_model(
                model,
                sampler_hmc,
                [1, 0.9, 0.75, 0.5, 0.25, 0.1],
                swap_strategy=MCMCTempering.NonReversibleSwap(),
                num_iterations=num_iterations,
                swap_every=10,
                adapt=false,
                mean_swap_rate_bound=0,
                init_params=copy(init_params),
                param_names=param_names,
                progress=false
            )
            map_parameters!(b, chain_tempered)

            # TODO: Make it not broken, i.e. produce reasonable results.
            compare_chains(chain_hmc, chain_tempered, atol=0.1, compare_std=false, compare_ess=false, isbroken=true)
        end
        
        @testset "AdvancedMH.jl" begin
            num_iterations = 100_000
            d = LogDensityProblems.dimension(model)

            # Set up MALA sampler.
            σ² = 1e-2
            sampler_mh = MALA(∇ -> MvNormal(σ² * ∇, 2σ² * I))

            # Sample using MALA.
            samples_mh = AbstractMCMC.sample(
                model, sampler_mh, num_iterations;
                init_params=copy(init_params), progress=false
            )
            chain_mh = AbstractMCMC.bundle_samples(
                samples_mh, MCMCTempering.maybe_wrap_model(model), sampler_mh, samples_mh[1], MCMCChains.Chains;
                param_names=param_names
            )
            map_parameters!(b, chain_mh)

            # Sample using tempered MALA.
            chain_tempered = test_and_sample_model(
                model,
                sampler_mh,
                [1, 0.9, 0.75, 0.5, 0.25, 0.1],
                swap_strategy=MCMCTempering.SingleSwap(),
                num_iterations=num_iterations,
                swap_every=2,
                adapt=false,
                mean_swap_rate_bound=0,
                init_params=copy(init_params),
                param_names=param_names
            )
            map_parameters!(b, chain_tempered)

            # TODO: Make it not broken, i.e. produce reasonable results.
            compare_chains(chain_mh, chain_tempered, atol=0.1, compare_std=false, compare_ess=false, isbroken=true)
        end
    end
end
