| :warning: WARNING          |
|:---------------------------|

This package is currently under development and non-functional. We anticipate the package will in working order by the end of summer 2022. Details below are subject to change.

# MCMCTempering.jl

MCMCTempering.jl implements simulated and parallel tempering, two methods for sampling from complex or multi-modal posteriors. These algorithms use temperature scheduling to flatten out the target distribution, making it easier to sample from.


## Tutorial: Using MCMCTempering

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


## Tutorial: Supporting MCMCTempering

This package can easily be extended to support any sampler following the lightweight `AbstractMCMC` interface. This tutorial uses `AdvancedHMC` as an example of how to support `MCMCTempering`.

### First Steps

1. Create a new file called `tempering.jl` which will contain all the functions required by `MCMCTempering`. 
2. Add `MCMCTempering` as a dependency for your package.
3. Add this code to your package:

```julia
##### ...
import MCMCTempering
include("tempering.jl")
export Joint, TemperedJoint, make_tempered_model, make_tempered_loglikelihood, get_params, step
##### ...
```

Now we need to add these functions to `tempering.jl`.


### Dialing up the temperature

First, we need to be able to call the log-likelihood after multiplying by an inverse temperature `β`. To do this we define the `make_tempered_model` function, which returns an instance of the relevant model type. For `AdvancedHMC` this is a `DifferentiableDensityModel`, but this should be whatever model type your sampler expects. Note that we only want to multiply the log-likelihood by `β`, not the log-prior; to do this, we define two callabale structs, `Joint` and `TemperedJoint`, each of which contains a log-prior and log-likelihood (and a temperature constant `β` for `TemperedJoint`). We need this so we can pass only a joint density function to `AdvancedHMC`, instead of the two components we require for tempering. These `Joint` structs let us return the log-joint as expected when we call  `model`'s density:

```julia
struct Joint{Tℓprior, Tℓll} <: Function
    ℓprior      :: Tℓprior
    ℓlikelihood :: Tℓll
end

function (joint::Joint)(θ)
    return joint.ℓprior(θ) .+ joint.ℓlikelihood(θ)
end


struct TemperedJoint{Tℓprior, Tℓll, T<:Real} <: Function
    ℓprior      :: Tℓprior
    ℓlikelihood :: Tℓll
    β           :: Real
end

function (tj::TemperedJoint)(θ)
    return tj.ℓprior(θ) .+ (tj.ℓlikelihood(θ) .* tj.β)
end


function MCMCTempering.make_tempered_model(
    model::DifferentiableDensityModel,
    β::Real
)
    ℓπ_β = TemperedJoint(model.ℓπ.ℓprior, model.ℓπ.ℓlikelihood, β)
    ∂ℓπ∂θ_β = TemperedJoint(model.∂ℓπ∂θ.ℓprior, model.∂ℓπ∂θ.ℓlikelihood, β)
    model_β = DifferentiableDensityModel(ℓπ_β, ∂ℓπ∂θ_β)
    return model_β
end
```

This gives us everything we need for `MCMCTempering` to adjust the temperature of the log-likelihood between steps.


### Carrying out temperature swap steps

For the tempering specific "swap steps" between pairs of chains' temperature levels, we need a way to temper the log-likelihood of the model without changing the model itself. To do this we implement the `make_tempered_loglikelihood` function, which takes a `model` and temperature `β` as arguments before returning a tempered loglikelihood function `logπ(z)`.

```julia
function MCMCTempering.make_tempered_loglikelihood(
    model::DifferentiableDensityModel,
    β::Real
)
    function logπ(z)
        return model.ℓπ.ℓlikelihood(z) * β
    end
    return logπ
end
```

Now we need access to the current proposed parameter values. This should be a simple getter function that returns the parameter vector `θ`:

```julia
function MCMCTempering.get_params(trans::Transition)
    return trans.z.θ
end
```

You can also choose to implement a function called `get_tempered_loglikelihoods_and_params` that returns the densities and parameters for the `k`th and `k+1`th chains. The code below is the default "fallback" implementation of this function, which should work for most cases:

```julia
function get_tempered_loglikelihoods_and_params(
    model,
    sampler::AbstractMCMC.AbstractSampler,
    states,
    k::Integer,
    Δ::Vector{Real},
    Δ_state::Vector{<:Integer}
)
    
    logπk = make_tempered_loglikelihood(model, Δ[Δ_state[k]])
    logπkp1 = make_tempered_loglikelihood(model, Δ[Δ_state[k + 1]])
    
    θk = get_params(states[k][1])
    θkp1 = get_params(states[k + 1][1])
    
    return logπk, logπkp1, θk, θkp1
end
```

Note that sometimes we do need to override this functionality. This is necessary in `Turing.jl`, for example, where the `sampler` and `VarInfo` are required to access the density and parameters:

```julia
function MCMCTempering.get_tempered_loglikelihoods_and_params(
    model::Model,
    sampler::Sampler{<:TemperedAlgorithm},
    states,
    k::Integer,
    Δ::Vector{Real},
    Δ_state::Vector{<:Integer}
)

    logπk = MCMCTempering.make_tempered_loglikelihood(model, Δ[Δ_state[k]], sampler, get_vi(states[k][2]))
    logπkp1 = MCMCTempering.make_tempered_loglikelihood(model, Δ[Δ_state[k + 1]], sampler, get_vi(states[k + 1][2]))
    
    θk = MCMCTempering.get_params(states[k][2], sampler)
    θkp1 = MCMCTempering.get_params(states[k + 1][2], sampler)
    
    return logπk, logπkp1, θk, θkp1
end


function MCMCTempering.make_tempered_loglikelihood(model::Model, β::Real, sampler::DynamicPPL.Sampler, varinfo_init::DynamicPPL.VarInfo)
    
    function logπ(z)
        varinfo = DynamicPPL.VarInfo(varinfo_init, sampler, z)
        model(varinfo)
        return DynamicPPL.getlogp(varinfo) * β
    end

    return logπ
end

get_vi(state::Union{HMCState,GibbsState,EmceeState,SMCState}) = state.vi
get_vi(vi::DynamicPPL.VarInfo) = vi

MCMCTempering.get_params(state, sampler::DynamicPPL.Sampler) = get_vi(state)[sampler]
```

This completes the tutorial.
