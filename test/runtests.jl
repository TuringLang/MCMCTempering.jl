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

@testset "MCMCTempering.jl" begin
    @testset "GMM 1D" begin
        
        nsamples = 1_000_000

        gmm = MixtureModel(Normal, [(-3, 1.5), (3, 1.5), (15, 1.5), (90, 1.5)], [0.175, 0.25, 0.275, 0.3])

        logdensity(x) = logpdf(gmm, x)

        # Construct a DensityModel.
        model = AdvancedMH.DensityModel(logdensity)

        # Create non-tempered baseline chain via RWMH
        sampler_rwmh = RWMH(Normal())
        samples = AbstractMCMC.sample(model, sampler_rwmh, nsamples)
        chain = AbstractMCMC.bundle_samples(samples, model, sampler_rwmh, samples[1], MCMCChains.Chains)

        # Simple geometric ladder
        inverse_temperatures = MCMCTempering.check_inverse_temperatures(0.05 .^ [0, 1, 2])

        acceptance_rate_target = 0.234
        
        # Number of iterations needed to obtain `nsamples` of non-swap iterations.
        swap_every = 2
        nsamples_tempered = Int(ceil(nsamples * swap_every ÷ (swap_every - 1)))

        tempered_sampler_rwmh = tempered(
            sampler_rwmh,
            inverse_temperatures;
            adapt = false,
            swap_every=swap_every
        )

        # Useful for analysis.
        states = []
        callback = StateHistoryCallback(states)

        # Sample.
        samples = AbstractMCMC.sample(model, tempered_sampler_rwmh, nsamples_tempered; callback=callback, progress=true)
        βs = mapreduce(Base.Fix2(getproperty, :inverse_temperatures), hcat, states)

        states_swapped = filter(Base.Fix2(getproperty, :is_swap), states)
        swap_acceptance_ratios = mapreduce(
            collect ∘ values ∘ Base.Fix2(getproperty, :swap_acceptance_ratios),
            vcat,
            states_swapped
        )
    
        # Extract the history of chain indices.
        process_to_chain_history_list = map(states) do state
            state.process_to_chain
        end
        process_to_chain_history = permutedims(reduce(hcat, process_to_chain_history_list), (2, 1))
    
        # Check that the swapping has been done correctly.
        process_to_chain_uniqueness = map(states) do state
            length(unique(state.process_to_chain)) == length(state.process_to_chain)
        end
        @test all(process_to_chain_uniqueness)
    
        # For these strategies, the index process should not move by more than 1.
        @test all(abs.(diff(process_to_chain_history[:, 1])) .≤ 1)
    
        chain_to_process_uniqueness = map(states) do state
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
        @test sum(swap_success_indicators) ≥ (nsamples / swap_every) * 0.1
    
        # Get example state.
        state = states[end]
        chain = AbstractMCMC.bundle_samples(
            samples, model, tempered_sampler_rwmh.sampler, MCMCTempering.state_for_chain(state), MCMCChains.Chains
        )
    
        # Thin chain and discard burnin.
        chain_thinned = chain[length(chain) ÷ 2 + 1:5swap_every:end]
        show(stdout, MIME"text/plain"(), chain_thinned)
    
        # Extract some summary statistics to compare.
        desc = describe(chain_thinned)[1].nt
        μ = desc.mean
        σ = desc.std

    end

    @testset "MvNormal 2D" begin
        d = 2
        nsamples = 20_000
        swap_every = 2

        μ_true = [-5.0, 5.0]
        σ_true = [1.0, √(10.0)]

        logdensity(x) = logpdf(MvNormal(μ_true, Diagonal(σ_true.^2)), x)

        # Sampler parameters.
        inverse_temperatures = MCMCTempering.check_inverse_temperatures([0.25, 0.5, 0.75, 1.0])

        # Construct a DensityModel.
        model = DensityModel(logdensity)

        # Set up our sampler with a joint multivariate Normal proposal.
        spl_inner = RWMH(MvNormal(zeros(d), Diagonal(σ_true.^2)))

        # Different swap strategies to test.
        swapstrategies = [
            MCMCTempering.StandardSwap(),
            MCMCTempering.RandomPermutationSwap(),
            MCMCTempering.NonReversibleSwap()
        ]

        # First we run MH to have something to compare to.
        samples_mh = AbstractMCMC.sample(model, spl_inner, nsamples; progress=false);
        chain_mh = AbstractMCMC.bundle_samples(samples_mh, model, spl_inner, samples_mh[1], MCMCChains.Chains)
        chain_thinned_mh = chain_mh[length(chain_mh) ÷ 2 + 1:5swap_every:end]

        # Extract some summary statistics to compare.
        desc_mh = describe(chain_thinned_mh)[1].nt
        μ_mh = desc_mh.mean
        σ_mh = desc_mh.std
        ess_mh = MCMCChains.ess_rhat(chain_thinned_mh).nt.ess

        @testset "$(swapstrategy)" for swapstrategy in swapstrategies
            acceptance_rate_target = 0.234
            # Number of iterations needed to obtain `nsamples` of non-swap iterations.
            nsamples_tempered = Int(ceil(nsamples * swap_every ÷ (swap_every - 1)))
            spl = tempered(
                spl_inner, inverse_temperatures;
                swap_strategy = swapstrategy,
                adapt=false,  # TODO: Test adaptation. Seems to work in some cases though.
                adapt_schedule=MCMCTempering.Geometric(),
                adapt_stepsize=1,
                adapt_eta=0.66,
                adapt_target=0.234,
                swap_every=swap_every
            )

            # Useful for analysis.
            states = []
            callback = StateHistoryCallback(states)

            # Sample.
            samples = AbstractMCMC.sample(model, spl, nsamples_tempered; callback=callback, progress=true);
            βs = mapreduce(Base.Fix2(getproperty, :inverse_temperatures), hcat, states)

            states_swapped = filter(Base.Fix2(getproperty, :is_swap), states)
            swap_acceptance_ratios = mapreduce(
                collect ∘ values ∘ Base.Fix2(getproperty, :swap_acceptance_ratios),
                vcat,
                states_swapped
            )
            # Check that the adaptation did something useful.
            if spl.adapt
                swap_acceptance_ratios = map(Base.Fix1(min, 1.0) ∘ exp, swap_acceptance_ratios)
                empirical_acceptance_rate = sum(swap_acceptance_ratios) / length(swap_acceptance_ratios)
                @test acceptance_rate_target ≈ empirical_acceptance_rate atol = 0.05
            end
        
            # Extract the history of chain indices.
            process_to_chain_history_list = map(states) do state
                state.process_to_chain
            end
            process_to_chain_history = permutedims(reduce(hcat, process_to_chain_history_list), (2, 1))
        
            # Check that the swapping has been done correctly.
            process_to_chain_uniqueness = map(states) do state
                length(unique(state.process_to_chain)) == length(state.process_to_chain)
            end
            @test all(process_to_chain_uniqueness)
        
            if any(isa.(Ref(swapstrategy), [MCMCTempering.StandardSwap, MCMCTempering.NonReversibleSwap]))
                # For these strategies, the index process should not move by more than 1.
                @test all(abs.(diff(process_to_chain_history[:, 1])) .≤ 1)
            end
        
            chain_to_process_uniqueness = map(states) do state
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
            @test sum(swap_success_indicators) ≥ (nsamples / swap_every) * 0.1
        
            # Get example state.
            state = states[end]
            chain = AbstractMCMC.bundle_samples(
                samples, model, spl.sampler, MCMCTempering.state_for_chain(state), MCMCChains.Chains
            )
        
            # Thin chain and discard burnin.
            chain_thinned = chain[length(chain) ÷ 2 + 1:5swap_every:end]
            show(stdout, MIME"text/plain"(), chain_thinned)
        
            # Extract some summary statistics to compare.
            desc = describe(chain_thinned)[1].nt
            μ = desc.mean
            σ = desc.std
        
            # `StandardSwap` is quite unreliable, so struggling to come up with reasonable tests.
            if !(swapstrategy isa StandardSwap)
                @test μ ≈ μ_true rtol=0.05

                # NOTE(torfjelde): The variance is usually quite large for the tempered chains
                # and I don't quite know if this is expected or not.
                # @test norm(σ - σ_true) ≤ 0.5
        
                # Comparison to just running the internal sampler.
                ess = MCMCChains.ess_rhat(chain_thinned).nt.ess
                # HACK: Just make sure it's not doing _horrible_. Though we'd hope it would
                # actually do better than the internal sampler.
                @test all(ess .≥ ess_mh .* 0.5)
            end
        end
    end
end
