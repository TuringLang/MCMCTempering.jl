include("setup.jl")

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
- `steps_per_swap`: The number of iterations between each swap attempt. Defaults to `1`.
- `adapt`: Whether to adapt the sampler. Defaults to `false`.
- `adapt_target`: The target acceptance rate for the swaps. Defaults to `0.234`.
- `adapt_rtol`: The relative tolerance for the check of average swap acceptance rate and target swap acceptance rate. Defaults to `0.1`.
- `adapt_atol`: The absolute tolerance for the check of average swap acceptance rate and target swap acceptance rate. Defaults to `0.05`.
- `mean_swap_rate_bound`: A bound on the acceptance rate of swaps performed, e.g. if set to `0.1` and `compare_mean_swap_rate=≥` then at least 10% of attempted swaps should be accepted. Defaults to `0.1`.
- `compare_mean_swap_rate`: a binary function for comparing average swap rate against `mean_swap_rate_bound`. Defaults to `≥`.
- `initial_params`: The initial parameters to use for the sampler. Defaults to `nothing`.
- `param_names`: The names of the parameters in the chain; used to construct the resulting chain. Defaults to `missing`.
- `progress`: Whether to show a progress bar. Defaults to `false`.
"""
function test_and_sample_model(
    model,
    sampler,
    inverse_temperatures;
    swap_strategy=MCMCTempering.SingleSwap(),
    mean_swap_rate_bound=0.1,
    compare_mean_swap_rate=≥,
    num_iterations=2_000,
    steps_per_swap=1,
    adapt=false,
    adapt_target=0.234,
    adapt_rtol=0.1,
    adapt_atol=0.05,
    initial_params=nothing,
    param_names=missing,
    progress=false,
    minimum_roundtrips=nothing,
    rng=make_rng(),
    kwargs...
)
    # Make the tempered sampler.
    sampler_tempered = tempered(
        sampler,
        inverse_temperatures;
        swap_strategy=swap_strategy,
        steps_per_swap=steps_per_swap,
        adapt_target=adapt_target,
    )

    @test sampler_tempered.swapstrategy == swap_strategy
    @test MCMCTempering.swapsampler(sampler_tempered).strategy == swap_strategy

    # Store the states.
    states_tempered = []
    callback = StateHistoryCallback(states_tempered)

    # Sample.
    samples_tempered = AbstractMCMC.sample(
        rng, model, sampler_tempered, num_iterations;
        callback=callback, progress=progress, initial_params=initial_params,
        kwargs...
    )

    if !isnothing(minimum_roundtrips)
        # Make sure we've had at least some roundtrips.
        @test length(MCMCTempering.roundtrips(samples_tempered)) ≥ minimum_roundtrips
    end

    # Let's make sure the process ↔ chain mapping is valid.
    numtemps = MCMCTempering.numtemps(sampler_tempered)
    @test all(states_tempered) do state
        all(1:numtemps) do i
            # These two should be inverses of each other.
            MCMCTempering.process_to_chain(state, MCMCTempering.chain_to_process(state, i)) == i
        end
    end

    # Extract the states that were swapped.
    states_swapped = map(Base.Fix2(getproperty, :swapstate), states_tempered)
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
    process_to_chain_history_list = map(states_swapped) do state
        state.process_to_chain
    end
    process_to_chain_history = permutedims(reduce(hcat, process_to_chain_history_list), (2, 1))

    # Check that the swapping has been done correctly.
    process_to_chain_uniqueness = map(states_swapped) do state
        length(unique(state.process_to_chain)) == length(state.process_to_chain)
    end
    @test all(process_to_chain_uniqueness)

    # For every strategy except `RandomSwap`, the index process should not move by more than 1.
    if !(swap_strategy isa Union{MCMCTempering.SingleRandomSwap,MCMCTempering.RandomSwap})
        @test all(abs.(diff(process_to_chain_history[:, 1])) .≤ 1)
    end

    chain_to_process_uniqueness = map(states_swapped) do state
        length(unique(state.chain_to_process)) == length(state.chain_to_process)
    end
    @test all(chain_to_process_uniqueness)

    # Compare the tempered sampler to the untempered sampler.
    state_tempered = states_tempered[end]
    chain_tempered = AbstractMCMC.bundle_samples(
        # TODO: Just use the underlying chain?
        samples_tempered,
        MCMCTempering.maybe_wrap_model(model),
        sampler_tempered,
        state_tempered,
        MCMCChains.Chains;
        param_names=param_names
    )

    # Tests that we have at least swapped some times (say at least 10% of attempted swaps).
    swap_success_indicators = map(eachrow(diff(process_to_chain_history; dims=1))) do row
        # Some of the strategies performs multiple swaps in a swap-iteration,
        # but we want to count the number of iterations for which we had a successful swap,
        # i.e. only count non-zero elements in a row _once_. Hence the `min`.
        min(1, sum(abs, row))
    end

    num_nonswap_steps_taken = length(chain_tempered)
    @test num_nonswap_steps_taken == (num_iterations * steps_per_swap)
    @test compare_mean_swap_rate(
        sum(swap_success_indicators),
        (num_nonswap_steps_taken / steps_per_swap) * mean_swap_rate_bound
    )

    return chain_tempered
end

function compare_chains(
    chain::MCMCChains.Chains, chain_tempered::MCMCChains.Chains;
    atol=1e-6, rtol=1e-6,
    compare_ess=true,
    compare_ess_slack=0.5, # HACK: this is very low which is unnecessary in most cases, but it's too random
    isbroken=false
)
    mean = to_dict(MCMCChains.mean(chain))
    mean_tempered = to_dict(MCMCChains.mean(chain_tempered))

    # Compare the means.
    if isbroken
        @test_broken all(isapprox(mean[sym], mean_tempered[sym]; atol, rtol) for sym in keys(mean))
    else
        @test all(isapprox(mean[sym], mean_tempered[sym]; atol, rtol) for sym in keys(mean))
    end

    # Compare the ESS.
    if compare_ess
        ess = to_dict(MCMCChains.ess(chain))
        ess_tempered = to_dict(MCMCChains.ess(chain_tempered))
        @info "" ess ess_tempered
        if isbroken
            @test_broken all(ess_tempered[sym] ≥ ess[sym] * compare_ess_slack for sym in keys(ess))
        else
            @test all(ess_tempered[sym] ≥ ess[sym] * compare_ess_slack for sym in keys(ess))
        end
    end
end

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
        MCMCTempering.swap!(chain_to_process, process_to_chain, 1, 2)
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
        MCMCTempering.swap!(chain_to_process, process_to_chain, 2, 3)
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
        num_iterations = 5_000
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
            adapt=false,
            initial_params=[[0.0], [1000.0]],  # initialized far apart
            # At MOST 1% of swaps should be successful.
            mean_swap_rate_bound=0.01,
            compare_mean_swap_rate=≤,
        )
        # `atol` is fairly high because we haven't run this for "too" long. 
        @test mean(chain_tempered[:, 1, :]) ≈ 1 atol=0.3
    end

    @testset "GMM 1D" begin
        num_iterations = 1_000
        model = DistributionLogDensity(
            MixtureModel(Normal, [(-3, 1.5), (3, 1.5), (15, 1.5), (90, 1.5)], [0.175, 0.25, 0.275, 0.3])
        )

        # Setup non-tempered.
        sampler_rwmh = RWMH(MvNormal(0.1 * Diagonal(Ones(1))))

        # Simple geometric ladder
        inverse_temperatures = MCMCTempering.check_inverse_temperatures(0.95 .^ (0:20))

        # There shouldn't be any swaps between two 
        chain_tempered = test_and_sample_model(
            model,
            sampler_rwmh,
            inverse_temperatures,
            swap_strategy=MCMCTempering.NonReversibleSwap(),
            num_iterations=num_iterations,
            adapt=false,
            # At least 25% of swaps should be successful.
            mean_swap_rate_bound=0.25,
            compare_mean_swap_rate=≥,
            progress=false,
            # Make sure we have _some_ roundtrips.
            minimum_roundtrips=10,
        )

        # # Compare the chains.
        # compare_chains(chain, chain_tempered, atol=1e-1, compare_std=false, compare_ess=true)
    end

    @testset "MvNormal 2D with different swap strategies" begin
        d = 2
        num_iterations = 5_000

        μ_true = [-5.0, 5.0]
        σ_true = [1.0, √(10.0)]

        # Sampler parameters.
        inverse_temperatures = MCMCTempering.check_inverse_temperatures([0.25, 0.5, 0.75, 1.0])

        # Construct a DensityModel.
        model = DistributionLogDensity(MvNormal(μ_true, Diagonal(σ_true .^ 2)))

        # Set up our sampler with a joint multivariate Normal proposal.
        sampler = RWMH(MvNormal(zeros(d), Diagonal(σ_true .^ 2)))
        # Sample for the non-tempered model for comparison.
        samples = AbstractMCMC.sample(make_rng(), model, sampler, num_iterations; progress=false)
        chain = AbstractMCMC.bundle_samples(
            samples, MCMCTempering.maybe_wrap_model(model), sampler, samples[1], MCMCChains.Chains
        )

        # Different swap strategies to test.
        swapstrategies = [
            MCMCTempering.ReversibleSwap(),
            MCMCTempering.NonReversibleSwap(),
            MCMCTempering.SingleSwap(),
            MCMCTempering.SingleRandomSwap(),
            MCMCTempering.RandomSwap()
        ]

        @testset "$(swap_strategy)" for swap_strategy in swapstrategies
            chain_tempered = test_and_sample_model(
                model,
                sampler,
                inverse_temperatures,
                num_iterations=num_iterations,
                swap_strategy=swap_strategy,
                adapt=false,
                # Make sure we have _some_ roundtrips.
                minimum_roundtrips=10,
                rng=make_rng(),
            )

            # Some swap strategies are not great.
            ess_slack_ratio = if swap_strategy isa Union{MCMCTempering.SingleRandomSwap,MCMCTempering.SingleSwap}
                0.25
            else
                0.5
            end
            compare_chains(chain, chain_tempered, rtol=0.1, compare_ess=true, compare_ess_slack=ess_slack_ratio)
        end
    end

    @testset "Turing.jl" begin
        # Let's make a default seed we can `deepcopy` throughout to get reproducible results.
        seed = 42

        # And let's set the seed explicitly for reproducibility.
        Random.seed!(seed)

        # Instantiate model.
        DynamicPPL.@model function demo_model(x)
            s ~ Exponential()
            m ~ Normal(0, s)
            x .~ Normal(m, s)
        end
        xs_true = rand(Normal(2, 1), 100)
        model_dppl = demo_model(xs_true)

        # Move to unconstrained space.
        vi = DynamicPPL.VarInfo(model_dppl)
        # Move to unconstrained space.
        vi = DynamicPPL.link!!(vi, model_dppl)
        # Get some initial values in unconstrained space.
        initial_params = rand(length(vi[:]))
        # Get the parameter names.
        param_names = map(Symbol, DynamicPPL.TestUtils.varnames(model_dppl))
        # Get bijector so we can get back to unconstrained space afterwards.
        b = inverse(Turing.bijector(model_dppl))
        # Construct the `LogDensityFunction` which supports LogDensityProblems.jl-interface.
        model = ADgradient(:ForwardDiff, DynamicPPL.LogDensityFunction(model_dppl, vi))

        @testset "Tempering of models" begin
            beta = 0.5
            model_tempered = MCMCTempering.make_tempered_model(model, beta)
            @test logdensity(model_tempered, initial_params) ≈ beta * logdensity(model, initial_params)
            @test last(logdensity_and_gradient(model_tempered, initial_params)) ≈
                beta .* last(logdensity_and_gradient(model, initial_params))
        end

        @testset "AdvancedHMC.jl" begin
            num_iterations = 5_000

            # Set up HMC smpler.
            initial_ϵ = 0.1
            integrator = AdvancedHMC.Leapfrog(initial_ϵ)
            proposal = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
                integrator, AdvancedHMC.GeneralisedNoUTurn()
            ))
            metric = AdvancedHMC.DiagEuclideanMetric(LogDensityProblems.dimension(model))
            sampler_hmc = AdvancedHMC.HMCSampler(proposal, metric, AdvancedHMC.NoAdaptation())

            # Sample using HMC.
            samples_hmc = sample(
                make_rng(seed), model, sampler_hmc, num_iterations;
                n_adapts=0,  # FIXME(torfjelde): Remove once AHMC.jl has fixed.
                initial_params=copy(initial_params),
                progress=false
            )
            chain_hmc = AbstractMCMC.bundle_samples(
                samples_hmc, MCMCTempering.maybe_wrap_model(model), sampler_hmc, samples_hmc[1], MCMCChains.Chains;
                param_names=param_names,
            )
            map_parameters!(b, chain_hmc)

            # Make sure that we get the "same" result when only using the inverse temperature 1.
            sampler_tempered = MCMCTempering.TemperedSampler(sampler_hmc, [1])
            chain_tempered = sample(
                make_rng(seed), model, sampler_tempered, num_iterations;
                n_adapts=0,  # FIXME(torfjelde): Remove once AHMC.jl has fixed.
                initial_params=copy(initial_params),
                chain_type=MCMCChains.Chains,
                param_names=param_names,
                progress=false,
            )
            map_parameters!(b, chain_tempered)
            compare_chains(
                chain_hmc, chain_tempered;
                atol=0.2,
                compare_ess=true,
                isbroken=false
            )

            # Sample using tempered HMC.
            chain_tempered = test_and_sample_model(
                model,
                sampler_hmc,
                [1, 0.75, 0.5, 0.25, 0.1, 0.01],
                swap_strategy=MCMCTempering.ReversibleSwap(),
                num_iterations=num_iterations,
                adapt=false,
                mean_swap_rate_bound=0.1,
                initial_params=copy(initial_params),
                param_names=param_names,
                progress=false,
                n_adapts=0,  # FIXME(torfjelde): Remove once AHMC.jl has fixed.
                rng=make_rng(seed),
            )
            map_parameters!(b, chain_tempered)
            compare_chains(
                chain_hmc, chain_tempered;
                atol=0.3,
                compare_ess=true,
                isbroken=false,
            )
        end

        @testset "AdvancedMH.jl" begin
            num_iterations = 10_000
            d = LogDensityProblems.dimension(model)

            # Set up MALA sampler.
            σ² = 1e-2
            sampler_mh = MALA(∇ -> MvNormal(σ² * ∇, 2σ² * I))

            # Sample using MALA.
            chain_mh = AbstractMCMC.sample(
                make_rng(), model, sampler_mh, num_iterations;
                initial_params=copy(initial_params),
                progress=false,
                chain_type=MCMCChains.Chains,
                param_names=param_names,
            )
            map_parameters!(b, chain_mh)

            # Make sure that we get the "same" result when only using the inverse temperature 1.
            sampler_tempered = MCMCTempering.TemperedSampler(sampler_mh, [1])
            chain_tempered = sample(
                make_rng(), model, sampler_tempered, num_iterations;
                initial_params=copy(initial_params),
                chain_type=MCMCChains.Chains,
                param_names=param_names,
                progress=false,
            )
            map_parameters!(b, chain_tempered)
            compare_chains(
                chain_mh, chain_tempered;
                atol=0.2,
                compare_ess=true,
                isbroken=false,
            )

            # Sample using actual tempering.
            chain_tempered = test_and_sample_model(
                model,
                sampler_mh,
                [1, 0.9, 0.75, 0.5, 0.25, 0.1],
                swap_strategy=MCMCTempering.ReversibleSwap(),
                num_iterations=num_iterations,
                adapt=false,
                mean_swap_rate_bound=0.1,
                initial_params=copy(initial_params),
                param_names=param_names,
                rng=make_rng(),
            )
            map_parameters!(b, chain_tempered)
            
            # Need a large atol as MH is not great on its own
            compare_chains(chain_mh, chain_tempered, atol=0.2, compare_ess=true, isbroken=false)
        end
    end

    include("abstractmcmc.jl")
    include("simple_gaussian.jl")
end
