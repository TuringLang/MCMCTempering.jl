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

At every swap step taken, this strategy _deterministically_ traverses
first the odd chain indices, proposing swaps between neighbors, and
then in the _next_ swap step taken traverses even chain indices, proposing
swaps between neighbors.

See [^SYED19] for more on this approach, referred to as DEO in their paper.

# References
[^SYED19]: Syed, S., Bouchard-Côté, Alexandre, Deligiannidis, G., & Doucet, A., Non-reversible Parallel Tempering: A Scalable Highly Parallel MCMC Scheme, arXiv:1905.02939,  (2019).
"""
struct NonReversibleSwap <: AbstractSwapStrategy end

"""
    SingleSwap <: AbstractSwapStrategy

At every swap step taken, this strategy samples a single chain index
`i` and proposes a swap between chains `i` and `i + 1`.

This approach goes under a number of names, e.g. Parallel Tempering
(PT) MCMC and Replica-Exchange MCMC.[^PTPH05]

# References
[^PTPH05]: Earl, D. J., & Deem, M. W., Parallel tempering: theory, applications, and new perspectives, Physical Chemistry Chemical Physics, 7(23), 3910–3916 (2005).
"""
struct SingleSwap <: AbstractSwapStrategy end

"""
    SingleRandomSwap <: AbstractSwapStrategy

At every swap step taken, this strategy samples two chain indices
`i` and 'j' and proposes a swap between the two corresponding chains.

This approach is shown to be effective for certain models in [^1].

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

Mainly useful for debugging or observing each chain independently,
this overrides and disables all swapping functionality.
"""
struct NoSwap <: AbstractSwapStrategy end

"""
    swap!(chain_to_process, process_to_chain, i, j)

Swaps the `i`th and `j`th temperatures in place.
"""
function swap!(chain_to_process, process_to_chain, i, j)
    # TODO: Use BangBang's `@set!!` to also support tuples?
    # Extract the process index for each of the chains.
    process_for_chain_i, process_for_chain_j = chain_to_process[i], chain_to_process[j]

    # Switch the mapping of the `chain → process` map.
    # The temperature for the i-th chain will now be moved from its current process
    # to the process for the (j)-th chain, and vice versa.
    chain_to_process[i], chain_to_process[j] = process_for_chain_j, process_for_chain_i

    # Swap the mapping of the `process → chain` map.
    # The process that used to have the i-th chain, now has the (i+1)-th chain, and vice versa.
    process_to_chain[process_for_chain_i], process_to_chain[process_for_chain_j] = j, i
    return chain_to_process, process_to_chain
end


"""
    compute_tempered_logdensities(model, sampler, transition, transition_other, β)
    compute_tempered_logdensities(model, sampler, sampler_other, transition, transition_other, state, state_other, β, β_other)

Return `(logπ(transition, β), logπ(transition_other, β))` where `logπ(x, β)` denotes the
log-density for `model` with inverse-temperature `β`.

The default implementation extracts the parameters from the transitions using [`getparams`](@ref) 
and calls [`logdensity`](@ref) on the model returned from [`make_tempered_model`](@ref).
"""
function compute_tempered_logdensities(model, sampler, transition, transition_other, β)
    tempered_model = make_tempered_model(sampler, model, β)
    return (
        logdensity(tempered_model, getparams(tempered_model, transition)),
        logdensity(tempered_model, getparams(tempered_model, transition_other))
    )
end
function compute_tempered_logdensities(
    model, sampler, sampler_other, transition, transition_other, state, state_other, β, β_other
)
    return compute_tempered_logdensities(model, sampler, transition, transition_other, β)
end

function compute_logdensities(
    model::AbstractMCMC.AbstractModel,
    model_other::AbstractMCMC.AbstractModel,
    state,
    state_other,
)
    # TODO: Make use of `getparams_and_logprob` instead?
    return (
        logdensity(model, getparams(model, state)),
        logdensity(model, getparams(model_other, state_other))
    )
end

"""
    swap_acceptance_pt(logπi, logπj)

Calculates and returns the swap acceptance ratio for swapping the temperature
of two chains. Using tempered likelihoods `logπi` and `logπj` at the chains'
current state parameters.
"""
function swap_acceptance_pt(logπiθi, logπiθj, logπjθi, logπjθj)
    return (logπjθi + logπiθj) - (logπiθi + logπjθj)
end


"""
    swap_attempt(rng, model, sampler, state, i, j)

Attempt to swap the temperatures of two chains by tempering the densities and
calculating the swap acceptance ratio; then swapping if it is accepted.
"""
function swap_attempt(rng::Random.AbstractRNG, model::MultiModel, sampler::SwapSampler, state, i, j)
    # Extract the relevant transitions.
    state_i = state_for_chain(state, i)
    state_j = state_for_chain(state, j)
    # Evaluate logdensity for both parameters for each tempered density.
    # NOTE: `SwapSampler` should only be working with models ordered according to `ProcessOrder`,
    # never `ChainOrder`, hence why we have the below.
    model_i = model_for_chain(ProcessOrder(), sampler, model, state, i)
    model_j = model_for_chain(ProcessOrder(), sampler, model, state, j)
    logπiθi, logπiθj = compute_logdensities(model_i, model_j, state_i, state_j)
    logπjθj, logπjθi = compute_logdensities(model_j, model_i, state_j, state_i)

    # If the proposed temperature swap is accepted according `logα`,
    # swap the temperatures for future steps.
    logα = swap_acceptance_pt(logπiθi, logπiθj, logπjθi, logπjθj)
    should_swap = -Random.randexp(rng) ≤ logα
    if should_swap
        swap!(state.chain_to_process, state.process_to_chain, i, j)
    end

    # Keep track of the (log) acceptance ratios.
    state.swap_acceptance_ratios[i] = logα

    # TODO: Handle adaptation.
    return state
end

