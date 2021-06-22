"""
    swap_betas(chain_index, k)

Swaps the `k`th and `k + 1`th temperatures.
Use `sortperm()` to convert the `chain_index` to a `Δ_index` to be used in tempering moves.
"""
function swap_betas(chain_index, k)
    chain_index[k], chain_index[k + 1] = chain_index[k + 1], chain_index[k]
    return sortperm(chain_index), chain_index
end

function make_tempered_loglikelihood end
function get_params end


"""
    get_tempered_loglikelihoods_and_params(model, sampler, states, k, Δ, chain_index)

Temper the `model`'s density using the `k`th and `k + 1`th temperatures 
selected via `Δ` and `chain_index`. Then retrieve the parameters using the chains'
current transitions extracted from the collection of `states`.
"""
function get_tempered_loglikelihoods_and_params(
    model,
    sampler::AbstractMCMC.AbstractSampler,
    states,
    k::Integer,
    Δ::Vector{Real},
    chain_index::Vector{<:Integer}
)
    
    logπk = make_tempered_loglikelihood(model, Δ[k])
    logπkp1 = make_tempered_loglikelihood(model, Δ[k + 1])
    
    θk = get_params(states[chain_index[k]][1])
    θkp1 = get_params(states[chain_index[k + 1]][1])
    
    return logπk, logπkp1, θk, θkp1
end


"""
    swap_acceptance_pt(logπk, logπkp1, θk, θkp1)

Calculates and returns the swap acceptance ratio for swapping the temperature
of two chains. Using tempered likelihoods `logπk` and `logπkp1` at the chains'
current state parameters `θk` and `θkp1`.
"""
function swap_acceptance_pt(logπk, logπkp1, θk, θkp1)
    return min(
        1,
        exp(logπkp1(θk) + logπk(θkp1)) / exp(logπk(θk) + logπkp1(θkp1))
        # exp(abs(βk - βkp1) * abs(AdvancedMH.logdensity(model, samplek) - AdvancedMH.logdensity(model, samplekp1)))
    )
end


"""
    swap_attempt(model, sampler, states, k, Δ, Δ_index)

Attempt to swap the temperatures of two chains by tempering the densities and
calculating the swap acceptance ratio; then swapping if it is accepted.
"""
function swap_attempt(model, sampler, ts, k, adapt, n)
    
    logπk, logπkp1, θk, θkp1 = get_tempered_loglikelihoods_and_params(model, sampler, ts.states, k, ts.Δ, ts.chain_index)
    
    swap_ar = swap_acceptance_pt(logπk, logπkp1, θk, θkp1)
    U = rand(Distributions.Uniform(0, 1))

    # If the proposed temperature swap is accepted according to swap_ar and U, swap the temperatures for future steps
    if U ≤ swap_ar
        ts.Δ_index, ts.chain_index = swap_betas(ts.chain_index, k)
    end

    # Adaptation steps affects Ρ and Δ, as the Ρ is adapted before a new Δ is generated and returned
    if adapt
        ts.Ρ, ts.Δ = adapt_ladder(ts.Ρ, ts.Δ, k, swap_ar, n)
    end
    return ts
end