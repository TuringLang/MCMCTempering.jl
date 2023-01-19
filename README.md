| :warning: WARNING          |
|:---------------------------|

This package is currently under development and non-functional. Details below are subject to change.

# MCMCTempering.jl

MCMCTempering.jl implements simulated and parallel tempering, two methods for sampling from complex or multi-modal posteriors. These algorithms use temperature scheduling to flatten out the target distribution, making it easier to sample from.


## Using MCMCTempering

`MCMCTempering` stores temperature scheduling information in a special kind of `sampler`. We can temper a sampler by calling the `tempered` function on any `sampler` that supports `MCMCTempering`, which includes all the samplers in `AdvancedHMC` and `AdvancedMH`. Here's an example:

```julia
using MCMCTempering

const temperature_steps = 4

sampler = NUTS()
tempered_sampler = tempered(sampler, temperature_steps)

chain = sample(model, tempered_sampler, n_samples; discard_initial = n_adapts)
```

It's that easy! Increasing the number of steps will make sampling easier for the sampler by avoiding any sudden changes in the posterior, but it'll also make the sampling take longer.

Enjoy your smooth sampling from multimodal posteriors!


## Supporting MCMCTempering

This package can easily be extended to support any sampler following the lightweight `AbstractMCMC` interface.

### The simple way

`AbstractMCMC.step` returns two things: a `transition` representing the state of the Markov chain, and a `state` representing the full state of the sampler. These are both kept track of internally and used by MCMCTempering.jl, and MCMCTempering.jl just needs a tiny bit of information on how to interact with these (in particular the latter one).

First we need to implement `MCMCTempering.getparams(transition)` so `MCMCTempering` knows how to extract parameters from the state of the Markov chain. Maybe it looks something like:

```julia
MCMCTempering.getparams(transition::MyTransition) = transition.θ
```

If your `model` type already implements [`LogDensityProblems.jl`](https://github.com/tpapp/LogDensityProblems.jl), that's it; **you're done!**

If it doesn't, then you also need to implement the following two methods:

```julia
MCMCTempering.logdensity(model, x) = ...                # Compute the log-density of `model` at `x`.
MCMCTempering.make_tempered_logdensity(model, β) = ...  # Return a tempered `model` which can be passed to `logdensity`.
```

Once that's done, you're good to go!

### Improving performance

When we're proposing a swap between the Markov chain targeting `model_left` with some temperature `β_left` and the chain targeting `model_right` with temperature `β_right`, we need to compute the following quantities (with the current realizations denoted `x_left` and `x_right`):

```julia
logdensity(model_left, x_left)
logdensity(model_right, x_right)
logdensity(model_left, x_right)
logdensity(model_right, x_left)
```

which can be computationally expensive. 

_But_ often the `transition` contains not only the current realization but also the log-density at that realization. In the above case, that means that for the first two quantities, i.e. `logdensity(model_left, x_left)` and `logdensity(model_right, x_right)`, we can just extract these from the corresponding Markov chain states `transition_left` and `transition_right`!

To make use of such cached computations, one has to explicitly implement `MCMCTempering.compute_tempered_logdensities`:

```julia
help?> MCMCTempering.compute_tempered_logdensities
  compute_tempered_logdensities(model, sampler, transition, transition_other, β)
  compute_tempered_logdensities(model, sampler, sampler_other, transition, transition_other, state, state_other, β, β_other)

  Return (logπ(transition, β), logπ(transition_other, β)) where logπ(x, β) denotes the log-density for model with
  inverse-temperature β.

  The default implementation extracts the parameters from the transitions using getparams and calls logdensity on the model
  returned from make_tempered_model.
```

Here one can just extract the corresponding quantities instead of computing them, and thus avoiding two additional calls to `logdensity(model, x)`.

Temper away!
