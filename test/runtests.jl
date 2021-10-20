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
        swap_every_n = 2

        μ_true = [-1.0, 1.0]
        σ_true = [1.0, √(10.0)]

        function logdensity(x)
            logpdf(MvNormal(μ_true, Diagonal(σ_true.^2)), x)
        end

        # Sampler parameters.
        Δ = MCMCTempering.check_Δ(0.5:0.01:1.0)

        # Construct a DensityModel.
        model = DensityModel(logdensity)

        # Set up our sampler with a joint multivariate Normal proposal.
        spl_inner = RWMH(MvNormal(zeros(d), 1e-1I))

        swapstrategies = [
            MCMCTempering.StandardSwap(),
            MCMCTempering.RandomPermutationSwap(),
            MCMCTempering.NonReversibleSwap()
        ]

        @testset "$(swapstrategy)" for swapstrategy in swapstrategies
            spl = tempered(spl_inner, Δ, swapstrategy; adapt=false, N_swap=swap_every_n)

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

            # Get example state.
            state = states[end]
            chain = if spl isa MCMCTempering.TemperedSampler
                AbstractMCMC.bundle_samples(
                    samples, model, spl.internal_sampler, MCMCTempering.state_for_chain(state), MCMCChains.Chains
                )
            else
                AbstractMCMC.bundle_samples(samples, model, spl, state, MCMCChains.Chains)
            end;

            # Thin chain and discard burnin.
            chain_thinned = chain[length(chain) ÷ 2 + 1:5swap_every_n:end]
            show(stdout, MIME"text/plain"(), chain_thinned)

            # Extract some summary statistics to compare.
            desc = describe(chain_thinned)[1].nt
            μ = desc.mean
            σ = desc.std

            # HACK: These bounds are quite generous. We're swapping quite frequently here
            # so some of the strategies results in a rather large variance of the estimators
            # it seems.
            show(stdout, MIME"text/plain"(), norm(μ - μ_true))
            show(stdout, MIME"text/plain"(), norm(σ - σ_true))
            @test norm(μ - μ_true) ≤ 0.5
            @test norm(σ - σ_true) ≤ 0.5
        end
    end
end
