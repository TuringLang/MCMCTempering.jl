# Getting started

## Mixture of Gaussians

Suppose we have a mixture of Gaussians, e.g. something like

```@example gmm
using Distributions
target_distribution = MixtureModel(
    Normal,
    [(-3, 1.5), (3, 1.5), (20, 1.5)],  # parameters
    [0.5, 0.3, 0.2]                    # weights
)
```

This is a simple 1-dimensional distribution, so let's visualize it:

```@example gmm
using StatsPlots
figsize = (800, 400)
plot(target_distribution; components=false, label=nothing, size=figsize)
```

We can convert a `Distribution` from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) into something we can pass to `sample` for many different samplers by implementing the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface:

```@example gmm
using LogDensityProblems: LogDensityProblems

struct DistributionLogDensity{D}
    d::D
end

LogDensityProblems.logdensity(d::DistributionLogDensity, x) = loglikelihood(d.d, x)
LogDensityProblems.dimension(d::DistributionLogDensity) = length(d.d)
LogDensityProblems.capabilities(::Type{<:DistributionLogDensity}) = LogDensityProblems.LogDensityOrder{0}()

# Wrap our target distribution.
target_model = DistributionLogDensity(target_distribution)
```

Immediately one might reach for a standard sampler, e.g. a random-walk Metropolis-Hastings (RWMH) from [`AdvancedMH.jl`](https://github.com/TuringLang/AdvancedMH.jl) and start sampling using `sample`:

```@example gmm
using AdvancedMH, MCMCChains, LinearAlgebra

using StableRNGs
rng = StableRNG(42) # To ensure reproducbility across devices.

sampler = RWMH(MvNormal(zeros(1), I))
num_iterations = 10_000
chain = sample(
    rng,
    target_model, sampler, num_iterations;
    chain_type=MCMCChains.Chains,
    param_names=["x"]
)
```

```@example gmm
plot(chain; size=figsize)
```

This doesn't look quite like what we're expecting.

```@example gmm
plot(target_distribution; components=false, linewidth=2)
density!(chain)
plot!(size=figsize)
```

Notice how `chain` has zero probability mass in the left-most component of the mixture!

Let's instead try to use a _tempered_ version of `RWMH`. _But_ before we do that, we need to make sure that AdvancedMH.jl is compatible with MCMCTempering.jl.

To do that we need to implement two methods. First we need to tell MCMCTempering how to extract the parameters, and potentially the log-probabilities, from a `AdvancedMH.Transition`:

```@docs
MCMCTempering.getparams_and_logprob
```

And similarly, we need a way to _update_ the parameters and the log-probabilities of a `AdvancedMH.Transition`:

```@docs
MCMCTempering.setparams_and_logprob!!
```

Luckily, implementing these is quite easy:

```@example gmm
using MCMCTempering

MCMCTempering.getparams_and_logprob(transition::AdvancedMH.Transition) = transition.params, transition.lp
function MCMCTempering.setparams_and_logprob!!(transition::AdvancedMH.Transition, params, lp)
    return AdvancedMH.Transition(params, lp)
end
```

Now that this is done, we can wrap `sampler` in a [`MCMCTempering.TemperedSampler`](@ref)

```@example gmm
inverse_temperatures = 0.90 .^ (0:20)
sampler_tempered = TemperedSampler(sampler, inverse_temperatures)
```

aaaaand `sample`!

```@example gmm
chain_tempered = sample(
    rng, target_model, sampler_tempered, num_iterations;
    chain_type=MCMCChains.Chains,
    param_names=["x"]
)
```

Let's see how this looks

```@example gmm
plot(chain_tempered)
plot!(size=figsize)
```

```@example gmm
plot(target_distribution; components=false, linewidth=2)
density!(chain)
density!(chain_tempered)
plot!(size=figsize)
```

Neato; we've indeed captured the target distribution much better!

We can even inspect _all_ of the tempered chains if we so desire

```@example gmm
chain_tempered_all = sample(
    rng,
    target_model, sampler_tempered, num_iterations;
    chain_type=Vector{MCMCChains.Chains},  # Different!
    param_names=["x"]
);
```

```@example gmm
plot(target_distribution; components=false, linewidth=2)
density!(chain)
# Tempered ones.
for chain_tempered in chain_tempered_all[2:end]
    density!(chain_tempered, color="green", alpha=inv(sqrt(length(chain_tempered_all))))
end
density!(chain_tempered_all[1], color="green", size=figsize)
plot!(size=figsize)
```
