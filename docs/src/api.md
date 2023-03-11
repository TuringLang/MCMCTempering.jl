# API

## Temper samplers

```@docs
MCMCTempering.tempered
MCMCTempering.TemperedSampler
```

Under the hood, [`MCMCTempering.TemperedSampler`](@ref) is actually just a "fancy" representation of a composition (represented using a [`MCMCTempering.CompositionSampler`](@ref)) of a [`MultiSampler`](@ref) and a [`SwapSampler`](@ref).

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

## Swapping

Swapping is implemented using the somewhat special [`MCMCTempering.SwapSampler`](@ref)

```@docs
MCMCTempering.SwapSampler
MCMCTempering.swapstrategy
```

This is a rather special sampler because, unlike most other implementations of `AbstractMCMC.AbstractSampler`, this is not a valid sampler _on its own_; indeed it is _required_ that this is at least in a composition (see [`MCMCTempering.CompositionSampler`](@ref)) with some other sampler.

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

To alleviate this issue we also have the [`RepeatedSampler`](@ref):

```@docs
MCMCTempering.RepeatedSampler
```

In the case where [`saveall`](@ref) returns `false`, `step` for a [`MCMCTempering.RepeatedSampler`](@ref) simply returns the last transition and state; if it returns `true`, then the transition is of type [`MCMCTempering.SequentialTransitions`](@ref) and the state is of type [`MCMCTempering.SequentialStates`](@ref).

```@docs
MCMCTempering.SequentialTransitions
MCMCTempering.SequentialStates
```

### Multiple or product of samplers

```@docs
MCMCTempering.MultiSampler
```

where the tempered models are represented using a [`MCMCTempering.MultiModel`](@ref)

```@docs
MCMCTempering.MultiModel
```

```@docs
MCMCTempering.make_tempered_model
```
