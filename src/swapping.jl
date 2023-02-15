"""
    AbstractSwapStrategy

Represents a strategy for swapping between parallel chains.

A concrete subtype is expected to implement the method [`swap_step`](@ref).
"""
abstract type AbstractSwapStrategy end

"""
    ReversibleSwap <: AbstractSwapStrategy

Stochastically attempt either even- or odd-indexed swap moves between chains.

See [^SYED19] for more on this approach, referred to as SEO in their paper.

# References
[^SYED19]: Syed, S., Bouchard-Côté, Alexandre, Deligiannidis, G., & Doucet, A., Non-reversible Parallel Tempering: A Scalable Highly Parallel MCMC Scheme, arXiv:1905.02939,  (2019).
"""
struct ReversibleSwap <: AbstractSwapStrategy end

"""
    NonReversibleSwap <: AbstractSwapStrategy

At every swap step taken, this strategy _deterministically_ traverses first the
odd chain indices, proposing swaps between neighbors, and then in the _next_ swap step
taken traverses even chain indices, proposing swaps between neighbors.

See [^SYED19] for more on this approach, referred to as DEO in their paper.

# References
[^SYED19]: Syed, S., Bouchard-Côté, Alexandre, Deligiannidis, G., & Doucet, A., Non-reversible Parallel Tempering: A Scalable Highly Parallel MCMC Scheme, arXiv:1905.02939,  (2019).
"""
struct NonReversibleSwap <: AbstractSwapStrategy end

"""
    SingleSwap <: AbstractSwapStrategy

At every swap step taken, this strategy samples a single chain index `i` and proposes
a swap between chains `i` and `i + 1`.

This approach goes under a number of names, e.g. Parallel Tempering (PT) MCMC and Replica-Exchange MCMC.[^PTPH05]

# References
[^PTPH05]: Earl, D. J., & Deem, M. W., Parallel tempering: theory, applications, and new perspectives, Physical Chemistry Chemical Physics, 7(23), 3910–3916 (2005).
"""
struct SingleSwap <: AbstractSwapStrategy end

"""
    SingleRandomSwap <: AbstractSwapStrategy

At every swap step taken, this strategy samples two chain indices `i` and 'j' and proposes
a swap between the two chains corresponding to `i` and `j`.

This approach goes under a number of names, e.g. Parallel Tempering (PT) MCMC and Replica-Exchange MCMC.[^1]

# References
[^1]: Malcolm Sambridge, A Parallel Tempering algorithm for probabilistic sampling and multimodal optimization, Geophysical Journal International, Volume 196, Issue 1, January 2014, Pages 357–374, https://doi.org/10.1093/gji/ggt342
"""
struct SingleRandomSwap <: AbstractSwapStrategy end

"""
    RandomSwap <: AbstractSwapStrategy

This strategy randomly shuffles all the chain indices to produce
`floor(numptemps(sampler)/2)` pairs of random (not necessarily
neighbouring) chain indices to attempt to swap
"""
struct RandomSwap <: AbstractSwapStrategy end

"""
    NoSwap <: AbstractSwapStrategy

Mainly useful for debugging or observing each chain independently, this overrides and disables all swapping functionality.
"""
struct NoSwap <: AbstractSwapStrategy end

"""
    swap_betas!(chain_order, chain_to_inverse_temperature_map, i, j)

Swaps the `i`th and `j`th temperatures in place.
"""
function swap_betas!(chain_order, chain_to_inverse_temperature_map, i, j)
    chain_order[i], chain_order[j] = chain_order[j], chain_order[i]
    chain_to_inverse_temperature_map[chain_order[i]], chain_to_inverse_temperature_map[chain_order[j]] = (
        chain_to_inverse_temperature_map[chain_order[j]], chain_to_inverse_temperature_map[chain_order[i]]
    )
end

"""
    compute_tempered_logdensities(model, state, idx_i, idx_j)

Return all 4 realisations of `(logπ(params_{i,j}, β_{i,j})` where `logπ(params, β)` denotes the
log-density of `model` tempered by inverse-temperature `β`.

The default implementation extracts the parameters from the transitions using [`getparams`](@ref) 
and calls [`logdensity`](@ref) on the model returned from [`make_tempered_model`](@ref).
"""
function compute_tempered_logdensities(model, state, idx_i, idx_j)
    β_i, β_j = get_inverse_temperature(state, idx_i), get_inverse_temperature(state, idx_j)
    params_i, params_j = get_transition_params(state, idx_i), get_transition_params(state, idx_j)
    return vcat(
        compute_tempered_logdensity(model, params_i, params_j, β_i),
        compute_tempered_logdensity(model, params_j, params_i, β_j)
    )
end
function compute_tempered_logdensity(model, params_a, params_b, β)
    tempered_model = make_tempered_model(model, β)
    return [logdensity(tempered_model, params_a), logdensity(tempered_model, params_b)]
end

"""
    swap_acceptance_pt(logπk, logπj)

Calculates and returns the swap acceptance ratio for swapping the temperature
of two chains. Using tempered likelihoods `logπk` and `logπj` at the chains'
current state parameters.
"""
function swap_acceptance_pt(logπk_θi, logπk_θj, logπj_θi, logπj_θj)
    return (logπj_θi + logπk_θj) - (logπk_θi + logπj_θj)
end

"""
    swap_attempt(rng, model, sampler, state, k, adapt)

Attempt to swap the temperatures of two chains by tempering the densities and
calculating the swap acceptance ratio; then swapping if it is accepted.
"""
function swap_attempt(rng, model, state, i, j)

    # Get the indices of the two chains to swap.
    idx_i, idx_j = state.chain_order[i], state.chain_order[j]

    # Evaluate each chain's tempered logdensity for both chains' parameters.
    logπk_θi, logπk_θj, logπj_θj, logπj_θi = compute_tempered_logdensities(
        model, state, idx_i, idx_j
    )
    
    # If the proposed temperature swap is accepted according `logα`,
    # swap the temperatures for future steps.
    logα = swap_acceptance_pt(logπk_θi, logπk_θj, logπj_θi, logπj_θj)
    if -Random.randexp(rng) ≤ logα
        swap_betas!(state.chain_order, state.chain_to_inverse_temperature_map, i, j)
        state.is_swap[idx_i] = true
        state.is_swap[idx_j] = true
    end

    # Keep track of the (log) acceptance ratios.
    state.swap_acceptance_ratios[i] = logα

    # Adaptation steps affects `ρs` and `inverse_temperatures`, as the `ρs` is
    # adapted before a new `inverse_temperatures` is generated and returned.
    # TODO this needs to support a Matrix of ρs to support the instances when `i` and `j` are not corresponding to neighbouring temperatures
    if !isnothing(state.adaptation_states)
        ρs = adapt!!(
            state.adaptation_states, state.inverse_temperatures, i, min(one(logα), exp(logα))
        )
        @set! state.adaptation_states = ρs
        @set! state.inverse_temperatures = update_inverse_temperatures(ρs, state.inverse_temperatures)
    end
    return state
end
