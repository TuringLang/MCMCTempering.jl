"""
    StateHistoryCallback

Defines a callable which simply pushes the `state` onto the `states` container.!

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
struct StateHistoryCallback{A}
    states::A
end
StateHistoryCallback() = StateHistoryCallback(Any[])

function (cb::StateHistoryCallback)(rng, model, sampler, sample, state, i; kwargs...)
    push!(cb.states, state)
    return nothing
end
