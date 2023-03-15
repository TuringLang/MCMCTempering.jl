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

[^SYED19]: Syed, S., Bouchard-Côté, Alexandre, Deligiannidis, G., & Doucet, A., Non-reversible Parallel Tempering: A Scalable Highly Parallel MCMC Scheme, arXiv:1905.02939,  (2019).
"""
struct NonReversibleSwap <: AbstractSwapStrategy end

"""
    SingleSwap <: AbstractSwapStrategy

At every swap step taken, this strategy samples a single chain index
`i` and proposes a swap between chains `i` and `i + 1`.

This approach goes under a number of names, e.g. Parallel Tempering
(PT) MCMC and Replica-Exchange MCMC.[^PTPH05]

[^PTPH05]: Earl, D. J., & Deem, M. W., Parallel tempering: theory, applications, and new perspectives, Physical Chemistry Chemical Physics, 7(23), 3910–3916 (2005).
"""
struct SingleSwap <: AbstractSwapStrategy end

"""
    SingleRandomSwap <: AbstractSwapStrategy

At every swap step taken, this strategy samples two chain indices
`i` and 'j' and proposes a swap between the two corresponding chains.

This approach is shown to be effective for certain models in [^1].

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
    compute_logdensities(model[, model_other], state, state_other)

Return `(logdensity(model, state), logdensity(model, state_other))`.

The default implementation extracts the parameters from the transitions using [`getparams`](@ref).

`model_other` can be provided to allow specializations that might be more efficient
if we know that `state_other` is from `model_other`, e.g. in the case where the log-probability
field is already present in `state` and `state_other`, and the only difference between
`logdensity(model, state_other)` and `logdensity(model_other, state_other)` is an easily computable
factor, then this can be exploited instead of re-computing the log-densities for both.
"""
function compute_logdensities(
    model::AbstractMCMC.AbstractModel,
    state,
    state_other,
)
    # TODO: Make use of `getparams_and_logprob` instead? At least for the `(model, state)` pair?
    return (
        logdensity(model, getparams(model, state)),
        logdensity(model, getparams(model, state_other))
    )
end

function compute_logdensities(
    model::AbstractMCMC.AbstractModel,
    model_other::AbstractMCMC.AbstractModel,
    state,
    state_other,
)
    return compute_logdensities(model, state, state_other)
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

