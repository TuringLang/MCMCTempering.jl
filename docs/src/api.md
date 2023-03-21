# API

## Temper samplers

```@docs
MCMCTempering.tempered
MCMCTempering.TemperedSampler
```

Under the hood, [`MCMCTempering.TemperedSampler`](@ref) is actually just a "fancy" representation of a composition (represented using a [`MCMCTempering.CompositionSampler`](@ref)) of a [`MCMCTempering.MultiSampler`](@ref) and a [`MCMCTempering.SwapSampler`](@ref).

Roughly speaking, the implementation of `AbstractMCMC.step` for [`MCMCTempering.TemperedSampler`](@ref) is basically

```julia
# 1. Construct the tempered models.
multimodel = MultiModel([make_tempered_model(model, β) for β in tempered_sampler.chain_to_beta])
# 2. Construct the samplers (can be the same one repeated multiple times or different ones)
multisampler = MultiSampler([getsampler(tempered_sampler, i) for i = 1:numtemps])
# 3. Step targeting `multimodel` using a compositoin of `multisampler` and `swapsampler`.
AbstractMCMC.step(rng, multimodel, multisampler ∘ swapsampler, state; kwargs...)
```

which in this case is provided by repeated calls to [`MCMCTempering.make_tempered_model`](@ref).

```@docs
MCMCTempering.make_tempered_model
```

This should be overloaded if you have some custom model-type that does not support the LogDensityProblems.jl-interface. In the case where the model _does_ support the LogDensityProblems.jl-interface, then the following will automatically be constructed

```@docs
MCMCTempering.TemperedLogDensityProblem
```

In addition, for computation of the tempered logdensities, we have

```@docs
MCMCTempering.compute_logdensities
```

## Swapping

Swapping is implemented using the somewhat special [`MCMCTempering.SwapSampler`](@ref)

```@docs
MCMCTempering.SwapSampler
MCMCTempering.swapstrategy
```

!!! warning 
    This is a rather special sampler because, unlike most other implementations of `AbstractMCMC.AbstractSampler`, this is not a valid sampler _on its own_; for this to be sensible it needs to be part of composition (see [`MCMCTempering.CompositionSampler`](@ref)) with _at least_ one other type of (an actually valid) sampler.

### Different swap-strategies

A [`MCMCTempering.SwapSampler`](@ref) can be defined with different swapping strategies:

```@docs
MCMCTempering.AbstractSwapStrategy
MCMCTempering.ReversibleSwap
MCMCTempering.NonReversibleSwap
MCMCTempering.SingleSwap
MCMCTempering.SingleRandomSwap
MCMCTempering.RandomSwap
MCMCTempering.NoSwap
```

```@docs
MCMCTempering.swap_step
```

## Other samplers

```@docs
MCMCTempering.saveall
```

### Compositions of samplers
```@docs
MCMCTempering.CompositionSampler
```

This sampler also has its own transition- and state-type

```@docs
MCMCTempering.CompositionTransition
MCMCTempering.CompositionState
```

#### Repeated sampler / composition with itself

Large compositions can have unfortunate effects on the compilation times in Julia.

To alleviate this issue we also have the [`MCMCTempering.RepeatedSampler`](@ref):

```@docs
MCMCTempering.RepeatedSampler
```

In the case where [`MCMCTempering.saveall`](@ref) returns `false`, `step` for a [`MCMCTempering.RepeatedSampler`](@ref) simply returns the last transition and state; if it returns `true`, then the transition is of type [`MCMCTempering.SequentialTransitions`](@ref) and the state is of type [`MCMCTempering.SequentialStates`](@ref).

```@docs
MCMCTempering.SequentialTransitions
MCMCTempering.SequentialStates
```

This effectively allows you to specify whether or not the "intermediate" states should be kept or not.

!!! note
    You will rarely see [`MCMCTempering.SequentialTransitions`](@ref) and [`MCMCTempering.SequentialStates`](@ref) as a user because `AbstractMCMC.bundle_samples` has been overloaded to these to return the flattened representation, i.e. we "un-roll" the transitions in every [`MCMCTempering.SequentialTransitions`](@ref).

### Multiple or product of samplers

```@docs
MCMCTempering.MultiSampler
```

where the tempered models are represented using a [`MCMCTempering.MultiModel`](@ref)

```@docs
MCMCTempering.MultiModel
```

The `step` for a [`MCMCTempering.MultiSampler`](@ref) and a [`MCMCTempering.MultiModel`] is a transition of type [`MCMCTempering.MultipleTransitions`](@ref) and a state of type [`MCMCTempering.MultipleStates`](@ref)

```@docs
MCMCTempering.MultipleTransitions
MCMCTempering.MultipleStates
```
