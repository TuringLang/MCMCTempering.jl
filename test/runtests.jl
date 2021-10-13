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
        function logdensity(x)
            logpdf(MvNormal(ones(length(x)), I), x)
        end

        # Sampler parameters.
        Δ = MCMCTempering.check_Δ(vcat(0.25:0.1:0.9, 0.91:0.005:1.0))

        # Construct a DensityModel.
        model = DensityModel(logdensity)

        # Set up our sampler with a joint multivariate Normal proposal.
        spl_inner = RWMH(MvNormal(zeros(d), 1e-1I))
        spl = tempered(spl_inner, Δ, MCMCTempering.StandardSwap(); adapt=false, N_swap=2)

        # Useful for analysis.
        states = []
        callback = StateHistoryCallback(states)

        # Sample.
        samples = AbstractMCMC.sample(model, spl, nsamples; callback=callback);
        states

        # Extract the history of chain indices.
        Δ_index_history_list = map(states) do state
            state.Δ_index
        end
        Δ_index_history = permutedims(reduce(hcat, Δ_index_history_list), (2, 1))

        # Get example state.
        state = states[end]
        chain = if spl isa MCMCTempering.TemperedSampler
            AbstractMCMC.bundle_samples(
                samples, model, spl.internal_sampler, state.states[first(state.chain_index)][2], MCMCChains.Chains
            )
        else
            AbstractMCMC.bundle_samples(samples, model, spl, state, MCMCChains.Chains)
        end;

        
        μ = mean(chain[length(chain) ÷ 2 + 1:10:end]).nt.mean
        # HACK: This is quite a large threshold.
        @test norm(μ - ones(length(μ))) ≤ 2e-1
    end
end
