"""
    AbstractSwapStrategy

Represents a strategy for swapping between parallel chains.

A concrete subtype is expected to implement the method [`swap_step`](@ref).
"""
abstract type AbstractSwapStrategy end

"""
    StandardSwap <: AbstractSwapStrategy

At every swap step taken, this strategy samples a single chain index `i` and proposes
a swap between chains `i` and `i + 1`.

This approach goes under a number of names, e.g. Parallel Tempering (PT) MCMC and Replica-Exchange MCMC.[^PTPH05]

# References
[^PTPH05]: Earl, D. J., & Deem, M. W., Parallel tempering: theory, applications, and new perspectives, Physical Chemistry Chemical Physics, 7(23), 3910–3916 (2005).
"""
struct StandardSwap <: AbstractSwapStrategy end

"""
    RandomPermutationSwap <: AbstractSwapStrategy

At every swap step taken, this strategy randomly shuffles all the chain indices
and then iterates through them, proposing swaps for neighboring chains.
"""
struct RandomPermutationSwap <: AbstractSwapStrategy end

"""
    NonReversibleSwap <: AbstractSwapStrategy

At every swap step taken, this strategy _deterministically_ traverses first the
odd chain indices, proposing swaps between neighbors, and then in the _next_ swap step
taken traverses even chain indices, proposing swaps between neighbors.

See [^SYED19] for more on this approach.

# References
[^SYED19]: Syed, S., Bouchard-Côté, Alexandre, Deligiannidis, G., & Doucet, A., Non-reversible Parallel Tempering: A Scalable Highly Parallel MCMC Scheme, arXiv:1905.02939,  (2019).
"""
struct NonReversibleSwap <: AbstractSwapStrategy end

"""
    NoSwap <: AbstractSwapStrategy

Mainly useful for debugging or observing each chain independently, this overrides and disables all swapping functionality.
"""
struct NoSwap <: AbstractSwapStrategy end

"""
    swap_betas!(chain_to_process, process_to_chain, k)

Swaps the `k`th and `k + 1`th temperatures in place.
"""
function swap_betas!(chain_order, chain_to_inverse_temperature_map, k)
    chain_order[k], chain_order[k + 1] = chain_order[k + 1], chain_order[k]
    chain_to_inverse_temperature_map[chain_order[k]], chain_to_inverse_temperature_map[chain_order[k + 1]] = chain_to_inverse_temperature_map[chain_order[k + 1]], chain_to_inverse_temperature_map[chain_order[k]]
end


"""
    compute_tempered_logdensities(model, sampler, transition, transition_other, β)
    compute_tempered_logdensities(model, sampler, sampler_other, transition, transition_other, state, state_other, β, β_other)

Return `(logπ(transition, β), logπ(transition_other, β))` where `logπ(x, β)` denotes the
log-density for `model` with inverse-temperature `β`.

The default implementation extracts the parameters from the transitions using [`getparams`](@ref) 
and calls [`logdensity`](@ref) on the model returned from [`make_tempered_model`](@ref).
"""
function compute_tempered_logdensities(model, params_k, params_kp1, β_k, β_kp1)
    return vcat(
        compute_tempered_logdensity(model, params_k, params_kp1, β_k),
        compute_tempered_logdensity(model, params_kp1, params_k, β_kp1)
    )
end
function compute_tempered_logdensity(model, params_a, params_b, β)
    tempered_model = make_tempered_model(model, β)
    return [logdensity(tempered_model, params_a), logdensity(tempered_model, params_b)]
end
# function compute_tempered_logdensities(
#     model, sampler, sampler_other, transition, transition_other, state, state_other, β, β_other
# )
#     return compute_tempered_logdensities(model, sampler, transition, transition_other, β)
# end

"""
    swap_acceptance_pt(logπk, logπkp1)

Calculates and returns the swap acceptance ratio for swapping the temperature
of two chains. Using tempered likelihoods `logπk` and `logπkp1` at the chains'
current state parameters.
"""
function swap_acceptance_pt(logπk_θk, logπk_θkp1, logπkp1_θk, logπkp1_θkp1)
    return (logπkp1_θk + logπk_θkp1) - (logπk_θk + logπkp1_θkp1)
end


"""
    swap_attempt(rng, model, sampler, state, k, adapt)

Attempt to swap the temperatures of two chains by tempering the densities and
calculating the swap acceptance ratio; then swapping if it is accepted.
"""
function swap_attempt(rng, model, state, k)
    # TODO: Allow arbitrary `k` rather than just `k + 1`.
    # Extract the relevant transitions.
    idx_k, idx_kp1 = state.chain_order[k], state.chain_order[k + 1]
    # Evaluate logdensity for both parameters for each tempered density.
    logπk_θk, logπk_θkp1, logπkp1_θkp1, logπkp1_θk = compute_tempered_logdensities(
        model,
        get_transition_params(state, idx_k), get_transition_params(state, idx_kp1),
        get_inverse_temperature(state, idx_k), get_inverse_temperature(state, idx_k)
    )
    
    # If the proposed temperature swap is accepted according `logα`,
    # swap the temperatures for future steps.
    logα = swap_acceptance_pt(logπk_θk, logπk_θkp1, logπkp1_θk, logπkp1_θkp1)
    if -Random.randexp(rng) ≤ logα
        swap_betas!(state.chain_order, state.chain_to_inverse_temperature_map, k)
    end

    # Keep track of the (log) acceptance ratios.
    @set! state.swap_acceptance_ratios[k] = logα

    # Adaptation steps affects `ρs` and `inverse_temperatures`, as the `ρs` is
    # adapted before a new `inverse_temperatures` is generated and returned.
    if !isnothing(state.adaptation_states)
        ρs = adapt!!(
            state.adaptation_states, state.inverse_temperatures, k, min(one(logα), exp(logα))
        )
        @set! state.adaptation_states = ρs
        @set! state.inverse_temperatures = update_inverse_temperatures(ρs, state.inverse_temperatures)
    end
    return state
end
