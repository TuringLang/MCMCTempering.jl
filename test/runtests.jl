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

        function logdensity(x)
            logpdf(MvNormal(ones(length(x)), I), x)
        end

        # Sampler parameters.
        Δ = MCMCTempering.check_Δ(vcat(0.25:0.1:0.9, 0.91:0.005:1.0))

        # Construct a DensityModel.
        model = DensityModel(logdensity)

        # Set up our sampler with a joint multivariate Normal proposal.
        spl_inner = RWMH(MvNormal(zeros(d), 1e-1I))

        swapstrategies = [
            MCMCTempering.StandardSwap(),
            MCMCTempering.RandomPermutationSwap(),
            MCMCTempering.NonReversibleSwap()
        ]
        @testset "$swapstrategy" for swapstrategy in swapstrategies
            swapstrategy = MCMCTempering.NonReversibleSwap()
            spl = tempered(spl_inner, Δ, swapstrategy; adapt=false, N_swap=swap_every_n)

            # TODO: Remove or make use of.
            # # Useful for analysis.
            # states = []
            # callback = StateHistoryCallback(states)
            callback = (args...; kwargs...) -> nothing

            # Sample.
            samples = AbstractMCMC.sample(model, spl, nsamples; callback=callback, progress=false);

            # # Extract the history of chain indices.
            # Δ_index_history_list = map(states) do state
            #     state.Δ_index
            # end
            # Δ_index_history = permutedims(reduce(hcat, Δ_index_history_list), (2, 1))

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
            chain_thinned = chain[length(chain) ÷ 2 + 1:swap_every_n:end]
            # Extract some summary statistics to compare.
            desc = describe(chain_thinned)[1].nt
            μ = desc.mean
            σ = desc.std

            # HACK: These bounds are quite generous. We're swapping quite frequently here
            # so some of the strategies results in a rather large variance of the estimators
            # it seems.
            @test norm(μ - ones(length(μ))) ≤ 2e-1
            @test norm(σ - ones(length(σ))) ≤ 3e-1

            # TODO: Add some tests so ensure that we are doing _some_ swapping?
        end
    end
end
