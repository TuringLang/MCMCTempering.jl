"""
    StateHistoryCallback

Defines a callable which pushes the `state` onto the `states` container.

Example usage when used with AbstractMCMC.jl:
```julia
# 1. Create empty container for state-history.
state_history = []
# 2. Sample.
AbstractMCMC.sample(model, sampler, N; callback=StateHistoryCallback(state_history))
# 3. Inspect states.
state_history
```
"""
struct StateHistoryCallback{A,F}
    states::A
    selector::F
end
StateHistoryCallback() = StateHistoryCallback(Any[])
function StateHistoryCallback(states, selector=deepcopy)
    return StateHistoryCallback{typeof(states), typeof(selector)}(states, selector)
end

function (cb::StateHistoryCallback)(rng, model, sampler, transition, state, iteration; kwargs...)
    push!(cb.states, cb.selector(state))
    return nothing
end


"""
    DistributionLogDensity(d)

Wraps a `Distribution` `d` and implements the LogDensityProblems.jl interface.
"""
struct DistributionLogDensity{D}
    d::D
end

LogDensityProblems.logdensity(d::DistributionLogDensity, x) = loglikelihood(d.d, x)
LogDensityProblems.dimension(d::DistributionLogDensity) = length(d.d)
LogDensityProblems.capabilities(::Type{<:DistributionLogDensity}) = LogDensityProblems.LogDensityOrder{0}()


"""
    map_parameters(f, chain)

Map the parameters of a `chain`.
"""
map_parameters(f, chain::MCMCChains.Chains) = map_parameters!(f, deepcopy(chain))


"""
    map_parameters!(f, chain)

Map the parameters of a `chain` in-place.
"""
function map_parameters!(f, chain::MCMCChains.Chains)
    params = chain.name_map.parameters
    for chain_idx = 1:size(chain, 3)
        for iter_idx = 1:size(chain, 1)
            x = f(vec(chain[iter_idx, params, chain_idx].value))
            for (par_idx, param) in enumerate(params)
                chain[iter_idx, param, chain_idx] = x[par_idx]
            end
        end
    end

    return chain
end
