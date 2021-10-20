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

The sampling of the chain index ensures reversibility/detailed balance is satisfied.

# References
[^PTPH05]: Earl, D. J., & Deem, M. W., Parallel tempering: theory, applications, and new perspectives, Physical Chemistry Chemical Physics, 7(23), 3910–3916 (2005).
"""
struct StandardSwap <: AbstractSwapStrategy end

"""
    RandomPermutationSwap <: AbstractSwapStrategy

At every swap step taken, this strategy randomly shuffles all the chain indices
and then iterates through them, proposing swaps for neighboring chains.

The shuffling of chain indices ensures reversibility/detailed balance is satisfied.
"""
struct RandomPermutationSwap <: AbstractSwapStrategy end


"""
    NonReversibleSwap <: AbstractSwapStrategy

At every swap step taken, this strategy _deterministically_ traverses first the
odd chain indices, proposing swaps between neighbors, and then in the _next_ swap step
taken traverses even chain indices, proposing swaps between neighbors.
"""
struct NonReversibleSwap <: AbstractSwapStrategy end

"""
    swap_betas!(chain_to_process, process_to_chain, k)

Swaps the `k`th and `k + 1`th temperatures.
"""
function swap_betas!(chain_to_process, process_to_chain, k)
    # TODO: Use BangBang's `@set!!` to also support tuples?
    # Extract the process index for each of the chains.
    process_for_chain_k, process_for_chain_kp1 = chain_to_process[k], chain_to_process[k + 1]

    # Switch the mapping of the `chain → process` map.
    # The temperature for the k-th chain will now be moved from its current process
    # to the process for the (k + 1)-th chain, and vice versa.
    chain_to_process[k], chain_to_process[k + 1] = process_for_chain_kp1, process_for_chain_k

    # Swap the mapping of the `process → chain` map.
    # The process that used to have the k-th chain, now has the (k+1)-th chain, and vice versa.
    process_to_chain[process_for_chain_k], process_to_chain[process_for_chain_kp1] = k + 1, k
    return chain_to_process, process_to_chain
end


"""
    compute_tempered_logdensities(model, sampler, transition, transition_other, β)

Return `(logπ(transition, β), logπ(transition_other, β))` where `logπ(x, β)` denotes the
log-density for `model` with inverse-temperature `β`.
"""
function compute_tempered_logdensities end

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
function swap_attempt(rng, model, sampler, state, k, adapt, total_steps)
    # Extract the relevant transitions.
    transitionk = transition_for_chain(state, k)
    transitionkp1 = transition_for_chain(state, k + 1)
    # Evaluate logdensity for both parameters for each tempered density.
    # NOTE: Here we want to propose swaps between the neighboring _chains_ not processes,
    # and so we get the `β` and `sampler` corresponding to the k-th and (k+1)-th chains.
    logπk_θk, logπk_θkp1 = compute_tempered_logdensities(
        model, sampler_for_chain(sampler, state, k), transitionk, transitionkp1, β_for_chain(state, k)
    )
    logπkp1_θkp1, logπkp1_θk = compute_tempered_logdensities(
        model, sampler_for_chain(sampler, state, k + 1), transitionkp1, transitionk, β_for_chain(state, k + 1)
    )
    
    # If the proposed temperature swap is accepted according `logα`,
    # swap the temperatures for future steps.
    logα = swap_acceptance_pt(logπk_θk, logπk_θkp1, logπkp1_θk, logπkp1_θkp1)
    if -Random.randexp(rng) ≤ logα
        swap_betas!(state.chain_to_process, state.process_to_chain, k)
    end

    # Adaptation steps affects Ρ and Δ, as the Ρ is adapted before a new Δ is generated and returned
    if adapt
        P, Δ = adapt_ladder(state.Ρ, state.Δ, k, min(one(logα), exp(logα)), total_steps)
        @set! state.Ρ = P
        @set! state.Δ = Δ
    end
    return state
end
