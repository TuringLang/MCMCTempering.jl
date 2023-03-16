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

### AdvancedHMC.jl

We also do this with AdvancedHMC.jl.

```@example gmm
using AdvancedHMC: AdvancedHMC
using ForwardDiff: ForwardDiff # for automatic differentation of the logdensity

# Creation of the sampler.
metric = AdvancedHMC.DiagEuclideanMetric(1)
integrator = AdvancedHMC.Leapfrog(0.1)
proposal = AdvancedHMC.StaticTrajectory(integrator, 8)
sampler = AdvancedHMC.HMCSampler(proposal, metric)
sampler_tempered = MCMCTempering.TemperedSampler(sampler, inverse_temperatures)

# Sample!
num_iterations = 5_000
chain = sample(
    rng,
    target_model, sampler, num_iterations;
    chain_type=MCMCChains.Chains,
    param_names=["x"],
)
plot(chain, size=figsize)
```

Then if we want to make it work with MCMCTempering, we define the same methods as before:

```@example gmm
# Provides a convenient way of "mutating" (read: reconstructing) types with different values
# for specified fields; see usage below.
using Setfield: Setfield

function MCMCTempering.getparams_and_logprob(state::AdvancedHMC.HMCState)
    t = state.transition
    return t.z.θ, t.z.ℓπ.value
end

function MCMCTempering.setparams_and_logprob!!(model, state::AdvancedHMC.HMCState, params, logprob)
    # NOTE: Need to recompute the gradient because it might be used in the next integration step.
    hamiltonian = AdvancedHMC.Hamiltonian(state.metric, model)
    return Setfield.@set state.transition.z = AdvancedHMC.phasepoint(
        hamiltonian, params, state.transition.z.r;
        ℓκ=state.transition.z.ℓκ
    )
end
```

And then, just as before, we can `sample`:

```@example gmm
chain_tempered_all = sample(
    StableRNG(42),
    target_model, sampler_tempered, num_iterations;
    chain_type=Vector{MCMCChains.Chains},
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


Works like a charm!

_But_ we're recomputing both the logdensity and the gradient of the logdensity upon every [`MCMCTempering.setparams_and_logprob!!`](@ref) above! This seems wholly unnecessary in the tempering case, since

```math
\pi_{\beta_1}(x) = \pi(x)^{\beta_1} = \big( \pi(x)^{\beta_2} \big)^{\beta_1 / \beta_2} = \pi_{\beta_2}^{\beta_1 / \beta_2}
```

i.e. if `model` in the above is tempered with ``\beta_1`` and the `params` are coming from a model with ``\beta_2``, we can could just compute it as

```julia
(β_1 / β_2) * logprob
```

and similarly for the gradient! Luckily, it's possible to tell MCMCTempering that this should be done by overloading the [`MCMCTempering.state_from`](@ref) method. In particular, we'll specify that when we're working with two models of type [`MCMCTempering.TemperedLogDensityProblem`](@ref) and two states of type `AdvancedHMC.HMCState`, then we can just re-use scale the logdensity and gradient computation from the [`MCMCTempering.state_from`](@ref) to get the quantities we want, thus avoiding unnecessary computations:

```@docs
MCMCTempering.state_from
```

```@example gmm
using AbstractMCMC: AbstractMCMC

function MCMCTempering.state_from(
    # AdvancedHMC.jl works with `LogDensityModel`, and by default `AbstractMCMC` will wrap
    # the input model with `LogDensityModel`, thus asusming it implements the
    # LogDensityProblems.jl-interface, by default.
    model::AbstractMCMC.LogDensityModel{<:MCMCTempering.TemperedLogDensityProblem},
    model_from::AbstractMCMC.LogDensityModel{<:MCMCTempering.TemperedLogDensityProblem},
    state::AdvancedHMC.HMCState,
    state_from::AdvancedHMC.HMCState,
)
    # We'll need the momentum and the kinetic energy from `ze.`
    z = state.transition.z
    # From this, we'll need everything else.
    z_from = state_from.transition.z
    params_from = z_from.θ
    logprob_from = z_from.ℓπ.value
    gradient_from = z_from.ℓπ.gradient

    # `logprob` is actually `β * actual_logprob`, and we want it to be `β_from * actual_logprob`, so
    # we can compute the "new" logprob as `(β_from / β) * logprob_from`.
    beta = model.logdensity.beta
    beta_from = model_from.logdensity.beta
    delta_beta = beta / beta_from
    logprob_new = delta_beta * logprob_from
    gradient_new = delta_beta .* gradient_from

    # Construct `PhasePoint`. Note that we keep `r` and `ℓκ` from the original state.
    return Setfield.@set state.transition.z = AdvancedHMC.PhasePoint(
        params_from,
        z.r,
        AdvancedHMC.DualValue(logprob_new, gradient_new),
        z.ℓκ
    )
end
```

!!! note
    For a general model we'd also have to do the same for [`MCMCTempering.compute_logdensities`](@ref) if we want to completely eliminate unnecessary computations, but for `AbstractMCMC.LogDensity{<:MCMCTempering.TemperedLogDensityProblem}` this is already implemented in MCMCTempering.

Now we can do the same but slightly faster:

```@example gmm
chain_tempered_all = sample(
    StableRNG(42),
    target_model, sampler_tempered, num_iterations;
    chain_type=Vector{MCMCChains.Chains},
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
