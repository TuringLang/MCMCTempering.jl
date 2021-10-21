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
    @testset "MvNormal 2D" begin
        d = 2
        nsamples = 20_000
        swap_every = 2

        μ_true = [-1.0, 1.0]
        σ_true = [1.0, √(10.0)]

        function logdensity(x)
            logpdf(MvNormal(μ_true, Diagonal(σ_true.^2)), x)
        end

        # Sampler parameters.
        inverse_temperatures = MCMCTempering.check_inverse_temperatures(vcat(0.5:0.05:0.7, 0.71:0.0025:1.0))

        # Construct a DensityModel.
        model = DensityModel(logdensity)

        # Set up our sampler with a joint multivariate Normal proposal.
        spl_inner = RWMH(MvNormal(zeros(d), 1e-1I))

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
            swapstrategy = swapstrategies[1]
            spl = tempered(spl_inner, inverse_temperatures, swapstrategy; adapt=false, swap_every=swap_every)

            # Useful for analysis.
            states = []
            callback = StateHistoryCallback(states)

            # Sample.
            samples = AbstractMCMC.sample(model, spl, nsamples; callback=callback, progress=false);

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
                # HACK: These bounds are quite generous. We're swapping quite frequently here
                # so some of the strategies results in a rather large variance of the estimators
                # it seems.
                @test norm(μ - μ_true) ≤ 0.5
                @test norm(σ - σ_true) ≤ 0.5

                # Comparison to just running the internal sampler.
                ess = MCMCChains.ess_rhat(chain_thinned).nt.ess
                @test all(ess .≥ ess_mh)
            end
        end
    end
end
