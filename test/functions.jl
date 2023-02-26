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
    num_iterations_tempered = Int(ceil(num_iterations * swap_every / (swap_every - 1)))

    # Make the tempered sampler.
    sampler_tempered = tempered(
        sampler,
        inverse_temperatures;
        swap_strategy=swap_strategy,
        swap_every=swap_every,
        adapt_target=adapt_target,
        model=model,
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

    # Let's make sure the process ↔ chain mapping is valid.
    numtemps = MCMCTempering.numtemps(sampler_tempered)
    for state in states_tempered
        for i = 1:numtemps
            # These two should be inverses of each other.
            @test MCMCTempering.process_to_chain(state, MCMCTempering.chain_to_process(state, i)) == i
        end
    end

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
    chain_tempered = AbstractMCMC.bundle_samples(
        samples_tempered[findall((!).(getproperty.(states_tempered, :is_swap)))],
        MCMCTempering.maybe_wrap_model(model),
        typeof(sampler) <: Function ? sampler(model) : sampler,
        samples_tempered[end],
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
        if isbroken
            @test_broken all(ess_tempered .≥ ess)
        else
            @test all(ess_tempered .≥ ess)
        end
    end
end

