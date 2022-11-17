using MCMCTempering
using Test
using Distributions
using AdvancedMH
using MCMCChains
using Bijectors
using LinearAlgebra
using AbstractMCMC

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
- `mean_swap_lower_bound`: A lower bound on the acceptance rate of swaps performed, e.g. if set to `0.1` then at least 10% of attempted swaps should be accepted. Defaults to `0.1`.
- `num_iterations`: The number of iterations to run the sampler for. Defaults to `2_000`.
- `swap_every`: The number of iterations between each swap attempt. Defaults to `2`.
- `adapt_target`: The target acceptance rate for the swaps. Defaults to `0.234`.
- `adapt_rtol`: The relative tolerance for the check of average swap acceptance rate and target swap acceptance rate. Defaults to `0.1`.
- `adapt_atol`: The absolute tolerance for the check of average swap acceptance rate and target swap acceptance rate. Defaults to `0.05`.
- `kwargs...`: Additional keyword arguments to pass to `MCMCTempering.tempered`.
"""
function test_and_sample_model(
    model,
    sampler,
    inverse_temperatures,
    swap_strategy=MCMCTempering.StandardSwap();
    mean_swap_rate_lower_bound=0.1,
    num_iterations=2_000,
    swap_every=2,
    adapt_target=0.234,
    adapt_rtol=0.1,
    adapt_atol=0.05,
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
        model, sampler_tempered, num_iterations_tempered; callback=callback, progress=true
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
    @test sum(swap_success_indicators) ≥ (num_iterations_tempered / swap_every) * mean_swap_rate_lower_bound

    # Compare the tempered sampler to the untempered sampler.
    state_tempered = states_tempered[end]
    chain_tempered = AbstractMCMC.bundle_samples(
        samples_tempered, model, sampler_tempered.sampler, MCMCTempering.state_for_chain(state_tempered), MCMCChains.Chains
    )
    # Only pick out the samples after swapping.
    # TODO: Remove this when no longer necessary.
    chain_tempered = chain_tempered[swap_every:swap_every:end]
    return chain_tempered
end

function compare_chains(
    chain::MCMCChains.Chains, chain_tempered::MCMCChains.Chains;
    atol=1e-6, rtol=1e-6,
    compare_std=true,
    compare_ess=true
)
    desc = describe(chain)[1].nt
    desc_tempered = describe(chain_tempered)[1].nt

    # Compare the means.
    @test desc.mean ≈ desc_tempered.mean atol = atol rtol = rtol

    # Compare the std. of the chains.
    if compare_std
        @test desc.std ≈ desc_tempered.std atol = atol rtol = rtol
    end

    # Compare the ESS.
    if compare_ess
        ess = MCMCChains.ess_rhat(chain).nt.ess
        ess_tempered = MCMCChains.ess_rhat(chain_tempered).nt.ess
        # HACK: Just make sure it's not doing _horrible_. Though we'd hope it would
        # actually do better than the internal sampler.
        @test all(ess .≥ ess_tempered .* 0.5)
    end
end


@testset "MCMCTempering.jl" begin
    @testset "GMM 1D" begin
        num_iterations = 100_000
        gmm = MixtureModel(Normal, [(-3, 1.5), (3, 1.5), (15, 1.5), (90, 1.5)], [0.175, 0.25, 0.275, 0.3])
        logdensity(x) = logpdf(gmm, x)

        # Setup non-tempered.
        model = AdvancedMH.DensityModel(logdensity)
        sampler_rwmh = RWMH(Normal())

        # Simple geometric ladder
        inverse_temperatures = MCMCTempering.check_inverse_temperatures(0.05 .^ [0, 1, 2])

        # Run the samplers.
        chain_tempered = test_and_sample_model(
            model,
            sampler_rwmh,
            [1.0, 0.5, 0.25, 0.125],
            num_iterations=num_iterations,
            swap_every=2,
            adapt=false,
        )

        # # Compare the chains.
        # compare_chains(chain, chain_tempered, atol=1e-1, compare_std=false, compare_ess=true)
    end

    @testset "MvNormal 2D" begin
        d = 2
        num_iterations = 20_000
        swap_every = 2

        μ_true = [-5.0, 5.0]
        σ_true = [1.0, √(10.0)]

        logdensity(x) = logpdf(MvNormal(μ_true, Diagonal(σ_true .^ 2)), x)

        # Sampler parameters.
        inverse_temperatures = MCMCTempering.check_inverse_temperatures([0.25, 0.5, 0.75, 1.0])

        # Construct a DensityModel.
        model = DensityModel(logdensity)

        # Set up our sampler with a joint multivariate Normal proposal.
        sampler = RWMH(MvNormal(zeros(d), Diagonal(σ_true .^ 2)))
        # Sample for the non-tempered model for comparison.
        samples = AbstractMCMC.sample(model, sampler, num_iterations)
        chain = AbstractMCMC.bundle_samples(samples, model, sampler, samples[1], MCMCChains.Chains)

        # Different swap strategies to test.
        swapstrategies = [
            MCMCTempering.StandardSwap(),
            MCMCTempering.RandomPermutationSwap(),
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
end
