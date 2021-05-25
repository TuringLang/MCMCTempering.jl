# MCMCTempering.jl

MCMCTempering.jl provides implementations of MCMC sampling algorithms such as simulated and parallel tempering, that are robust to multi-modal target distributions. These algorithms leverage temperature scheduling to flatten out the target distribution and allow sampling to move more freely around a target's complete state space to better explore its mass.


# Tutorial: Supporting MCMCTempering for an arbitrary sampler, namely AdvancedHMC

We offer support for `MCMCTempering`'s sampling functionality through extrogenous implementation of the required API components in the `Turing`, `AdvancedHMC` and `AdvancedMH` packages. This package has been built such that a minimal set of components are required to allow (a) sampler(s) to be wrapped by `MCMCTempering` and all of its accompanying functionality. Aside from the three packages we "natively" support, there is only a light base assumption that a package implements the interface offered in `AbstractMCMC` for samplers in order for it to also be supported by `MCMCTempering`. This is a fairly lax requirement given the lightweight nature of this package and we recommend inspecting its (minimal) interface before writing off `AbstractMCMC` and further, facilitating support of this package.

To illustrate this, we step through an example showing all of the work required in order to get `AdvancedHMC` working with `MCMCTempering`.

## Tempering a sampler

Firstly, observing the signature of the generic `sample` call exposed by `AbstractMCMC`, we see that we minimally require a `model`, a `sampler` and other args such as the number of samples `N` to return, etc. Given the aforementioned base assumption that your sampler should conform to `AbstractMCMC`'s `sample` and `step` structure, it is sufficient here to ensure these methods will be called as expected. In general, we carry through the tempering schedule and other information via the `sampler` as this can be presumed to be present at each step of a sampling routine; to do this in `AdvancedHMC`'s case, we must circumnavigate the internal definition of `AbstractMCMC.sample` (which requires a user to provide a `kernel`, `metric` and `adaptor`) to build a `sampler` object ourselves (in this case we want to temper the `HMCSampler` from `AdvancedHMC` which is a struct containing the aforementioned three components), this can be done as in this minimal working example based on standard usage of the `AdvancedHMC` package. These first lines are to setup sampling, as in any other standard usage of `AdvancedHMC`:

```julia
using AdvancedHMC, Distributions, ForwardDiff

# Choose parameter dimensionality and initial parameter value
D = 10; initial_θ = rand(D)
n_samples, n_adapts = 2_000, 1_000

# Define the target distribution
ℓprior(θ) = 0
ℓlikelihood(θ) = logpdf(MvNormal(zeros(D), ones(D)), θ)
∂ℓprior∂θ(θ) = ForwardDiff.gradient(ℓprior, θ)
∂ℓlikelihood∂θ(θ) = ForwardDiff.gradient(ℓlikelihood, θ)
model = DifferentiableDensityModel(
    Joint(ℓprior, ℓlikelihood),
    Joint(∂ℓprior∂θ, ∂ℓlikelihood∂θ)
)

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, model.ℓπ, model.∂ℓπ∂θ)
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
```

It is only after this step that we diverge from "standard usage" to pre-define a `HMCSampler` (note that this is done internally during the `AbstractMCMC.sample` function defined in `AdvancedHMC` anyway) and then wrap it in a call of `Tempered`, providing an integer number of tempering levels to use:

```julia
using MCMCTempering

sampler = HMCSampler(proposal, metric, adaptor)

chain = sample(model, Tempered(sampler, 4), n_samples; discard_initial = n_adapts)
```

So usage is fairly simple provided we stick with the expected `AbstractMCMC.sample` call arguments, to facilitate this usage of tempering we must next implement the minimal API described below.

## Stepping using the sampler and a tempered model

The first requirement for tempering is to be able to call the `model`'s log likelihood density function in product with an inverse temperature. For this we define the `make_tempered_model` function that returns an instance of the relevant model type; in `AdvancedHMC`'s case this is a `DifferentiableDensityModel`, but should be whatever model type your sampler expects, where the internals of the `model`'s log likelihood are adjusted according to an inverse temperature multiplier `β`:

```julia
function MCMCTempering.make_tempered_model(model::DifferentiableDensityModel, β::T) where {T<:AbstractFloat}
    ℓπ_β(θ) = model.ℓπ(θ) * β
    ∂ℓπ∂θ_β(θ) = model.∂ℓπ∂θ(θ) * β
    model = DifferentiableDensityModel(ℓπ_β, ∂ℓπ∂θ_β)
    return model
end
```

This is all that is required to ensure `MCMCTempering`'s functionality injects between each step and ensures each chain in our implementation of parallel tempering can step according to the correct inverse temperature `β`.

## Carrying out temperature swap steps

For the tempering specific "swap steps" between pairs of chains' temperature levels, we must first offer a way to temper the *density* of the model; for this we implement the `make_tempered_logπ` function which accepts the `model` and a temperature `β`; then it returns a function `logπ(z)` which is a transformation of the `model`'s log likelihood function:

```julia
function MCMCTempering.make_tempered_logπ(model::DifferentiableDensityModel, β::T) where {T<:AbstractFloat}
    function logπ(z)
        return model.ℓπ(z) * β
    end
    return logπ
end
```

Access to the current proposed parameter values is required, and this should be a relatively simple getter function accessing the current `state` of the sampler in most cases to return `θ`:

```julia
function MCMCTempering.get_θ(state::HMCState)
    return state.z.θ
end
```

Both of these parts should then be used in a function called `get_densities_and_θs` that returns the densities and parameters for the `k`th and `k+1`th chains, the interface is built in this way as the requirements for accessing the two aforementioned components can reasonably change, with some samplers being built such that they require state information, sampler information, model information etc. to access these properties, this allows for flexibility in implementation.

In this case, the implementation of `get_densities_and_θs` is relatively simple, in fact, the code below is the default "fallback" implementation of this function and so provided your sampler submits to this fairly standard set of arguments you do not need to implement this method, as is the case for `AdvancedHMC`:

```julia
function get_densities_and_θs(
    model,
    sampler::AbstractMCMC.AbstractSampler,
    states,
    k::Integer,
    Δ::Vector{T},
    Δ_state::Vector{<:Integer}
) where {T<:AbstractFloat}
    logπk = make_tempered_logπ(model, Δ[Δ_state[k]])
    logπkp1 = make_tempered_logπ(model, Δ[Δ_state[k + 1]])
    θk = get_θ(states[k][2])
    θkp1 = get_θ(states[k + 1][2])
    return logπk, logπkp1, θk, θkp1
end
```

In some cases we do need to override this functionality. This is necessary in `Turing.jl` for example where the `sampler` and `VarInfo` are required to access the density and parameters resulting in the following implementations:

```julia
function MCMCTempering.get_densities_and_θs(
    model::Model,
    sampler::Sampler{<:TemperedAlgorithm},
    states,
    k::Integer,
    Δ::Vector{T},
    Δ_state::Vector{<:Integer}
) where {T<:AbstractFloat}

    logπk = MCMCTempering.make_tempered_logπ(model, Δ[Δ_state[k]], sampler, get_vi(states[k][2]))
    logπkp1 = MCMCTempering.make_tempered_logπ(model, Δ[Δ_state[k + 1]], sampler, get_vi(states[k + 1][2]))
    
    θk = MCMCTempering.get_θ(states[k][2], sampler)
    θkp1 = MCMCTempering.get_θ(states[k + 1][2], sampler)
    
    return logπk, logπkp1, θk, θkp1
end


function MCMCTempering.make_tempered_logπ(model::Model, β::T, sampler::DynamicPPL.Sampler, varinfo_init::DynamicPPL.VarInfo) where {T<:AbstractFloat}
    
    function logπ(z)
        varinfo = DynamicPPL.VarInfo(varinfo_init, sampler, z)
        model(varinfo)
        return DynamicPPL.getlogp(varinfo) * β
    end

    return logπ
end

get_vi(state::Union{HMCState,GibbsState,EmceeState,SMCState}) = state.vi
get_vi(vi::DynamicPPL.VarInfo) = vi

MCMCTempering.get_θ(state, sampler::DynamicPPL.Sampler) = get_vi(state)[sampler]
```

At this point we have implemented all of the necessary components such that the first code block will run and we can temper `AdvancedHMC`'s samplers successfully.