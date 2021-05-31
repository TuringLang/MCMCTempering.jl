# MCMCTempering.jl

MCMCTempering.jl provides implementations of MCMC sampling algorithms such as simulated and parallel tempering, that are robust to multi-modal target distributions. These algorithms leverage temperature scheduling to flatten out the target distribution and allow sampling to move more freely around a target's complete state space to better explore its mass.


# Tutorial: Supporting MCMCTempering for an arbitrary sampler, namely AdvancedHMC

We offer support for `MCMCTempering`'s sampling functionality through extrogenous implementation of the required API components in the `Turing`, `AdvancedHMC` and `AdvancedMH` packages. This package has been built such that a minimal set of components are required to allow (a) sampler(s) to be wrapped by `MCMCTempering` and all of its accompanying functionality. Aside from the three packages we "natively" support, there is only a light base assumption that a package implements the interface offered in `AbstractMCMC` for samplers in order for it to also be supported by `MCMCTempering`. This is a fairly lax requirement given the lightweight nature of this package and we recommend inspecting its (minimal) interface before writing off `AbstractMCMC` and further, facilitating support of this package.

To illustrate this, we step through an example showing all of the work required in order to get `AdvancedHMC` working with `MCMCTempering`.

## First Steps

We should define a new file, `tempering.jl` is a reasonable name and the choice we made for all of our supported packages, to contain the `MCMCTempering` function implementations. All of the functions discussed below are to be written in this file, and then should be exported such that they can be accessed during sampling. To do this correctly, `MCMCTempering` should also be added as a project dependency to the sampling package in question. Then something along the lines below should be included such that the implemented API can be accessed by the samplers of the package during the calls to `MCMCTempering`'s functionality:

```julia
##### ...
import MCMCTempering
include("tempering.jl")
export Joint, TemperedJoint, make_tempered_model, make_tempered_loglikelihood, get_params
##### ...
```

With this in mind, we should populate `tempering.jl` with the aforementioned implementations. The first major consideration to deal with is how to carry through all of the required tempering information upon calling `sample` on our sampler.

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
∂ℓprior∂θ(θ) = (ℓprior(θ), ForwardDiff.gradient(ℓprior, θ))
∂ℓlikelihood∂θ(θ) = (ℓlikelihood(θ), ForwardDiff.gradient(ℓlikelihood, θ))
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

The first requirement for tempering is to be able to call the `model`'s **log-likelihood** density function in product with an inverse temperature. For this we define the `make_tempered_model` function that returns an instance of the relevant model type - in `AdvancedHMC`'s case this is a `DifferentiableDensityModel` (but this should of course be whatever model type your sampler expects) - where the internals of the `model`'s log-likelihood are adjusted according to an inverse temperature multiplier `β`. To achieve the desired functionality and act *only* on the log-likelihood whilst leaving the log-prior as is, we define two callabale structs `Joint` and `TemperedJoint` containing a log-prior and log-likelihood (and a temperature `β`). This is necessary so as to adhere correctly with `AdvancedHMC`'s expectation of a `model` that wraps only a **joint** density function rather than the two components we require for tempering, and the `Joint` structs allow us to by default return the value of the log-joint as expected uon calling the `model`'s density:

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

This is all that is required to ensure `MCMCTempering`'s functionality injects between each step and ensures each chain in our implementation of parallel tempering can step according to the correct inverse temperature `β`.

## Carrying out temperature swap steps

For the tempering specific "swap steps" between pairs of chains' temperature levels, we must similarly offer a way to temper the **log-likelihood** of the model independently from the model itself; for this we implement the `make_tempered_loglikelihood` function which accepts the `model` and a temperature `β`; then it returns a function `logπ(z)` which is a transformation of the `model`'s log-likelihood function contained in the aforementioned `Joint`:

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

Access to the current proposed parameter values is required, and this should be a relatively simple getter function, accessing the current `state` of the sampler in most cases, to then return `θ`:

```julia
function MCMCTempering.get_params(trans::Transition)
    return trans.z.θ
end
```

Both of these parts should then be used in a function called `get_tempered_loglikelihoods_and_params` that returns the densities and parameters for the `k`th and `k+1`th chains, the interface is built in this way as the requirements for accessing the two aforementioned components can reasonably change, with some samplers being built such that they require state information, sampler information, model information etc. to access these properties, this allows for flexibility in implementation.

In this case, the implementation of `get_tempered_loglikelihoods_and_params` is relatively simple, in fact, the code below is the default "fallback" implementation of this function and so provided your sampler submits to this fairly standard set of arguments you do not need to implement this method at all, as is the case for `AdvancedHMC`:

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

Note that in some cases we **do** need to override this functionality. This is necessary in `Turing.jl` for example where the `sampler` and `VarInfo` are required to access the density and parameters resulting in the following implementations:

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

At this point we have implemented all of the necessary components such that the first code block will run and we can temper `AdvancedHMC`'s samplers successfully.