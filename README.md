# MCMCTempering.jl

MCMCTempering.jl provides implementations of MCMC sampling algorithms such as simulated and parallel tempering, that are robust to multi-modal target distributions. These algorithms leverage temperature scheduling to flatten out the target distribution and allow sampling to move more freely around a target's complete state space to better explore its mass.


# Tutorial: Supporting MCMCTempering for an arbitrary sampler

We offer support through extrogenous implementation of the required API components in Turing, AdvancedHMC and AdvancedMH. This package has been built such that a minimal set of components are required in your package of samplers to allow them to be wrapped by MCMCTempering and all of its offered approaches. There is a base assumption that the sampler in question implements the interface offered in `AbstractMCMC`, this is a fairly lax requirement given the lightweight nature of this package and we recommend inspecting its (minimal) interface before writing off facilitating support of this package.

To illustrate this, we step through a relatively simple example in order to get `AdvancedHMC` working with MCMCTempering.

## Tempering a sampler

Firstly, we must observe the signature of the generic `sample` call exposed by `AbstractMCMC`, for this we require a `model`, a `sampler` and other args such as the number of samples `N`, whether or not to run chains in `parallel` etc. Given the aforementioned base assumption that your sampler should conform to `AbstractMCMC`'s `sample` and `step` structure, it is sufficient here to ensure these methods will be called as expected. In general, we carry through the tempering schedule and other information via the `sampler` as this is present in all cases; to do this in `AdvancedHMC`'s case, we must circumnavigate the internal definition of sample and build a `sampler` object ourselves (in this case we want to temper the `HMCSampler` from `AdvancedHMC`), this can be done like so:

```julia
using AdvancedHMC, Distributions, ForwardDiff, MCMCTempering

# Choose parameter dimensionality and initial parameter value
D = 10; initial_θ = rand(D)
n_samples, n_adapts = 2_000, 1_000

# Define the target distribution
ℓπ(θ) = logpdf(MvNormal(zeros(D), ones(D)), θ)
∂ℓπ∂θ = ForwardDiff
model = DifferentiableDensityModel(ℓπ, ∂ℓπ∂θ)

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

sampler = HMCSampler(proposal, metric, adaptor)

chain = sample(model, Tempered(sampler, 4), n_samples; discard_initial = n_adapts)
```

## Stepping using the sampler and a tempered model

Now we must provide a `make_tempered_model` implementation that returns an instance of the relevant model type, in `AdvancedHMC`'s case this is a `DifferentiableDensityModel` but should be whatever model type your sampler expects, where the internals of the model are adjusted according to an inverse temperature `β`:

```julia
function make_tempered_model(model::DifferentiableDensityModel, β::T) where {T<:AbstractFloat}
    ℓπ_β(θ) = model.ℓπ(θ) * β
    ∂ℓπ∂θ_β(θ) = model.∂ℓπ∂θ(θ) * β
    model = DifferentiableDensityModel(ℓπ_β, ∂ℓπ∂θ_β)
    return model
end
```

## Carrying out temperature swap steps

We must first offer a way to temper the *density* of the model, this is used during a temperature swap step, for this we implement `make_tempered_logπ` which accepts the `modeel` and a temperature `β` then returns a function `logπ(z)` which is a transformation of the `model`'s log likelihood function:

```julia
function make_tempered_logπ(model::DifferentiableDensityModel, β::T) where {T<:AbstractFloat}
    function logπ(z)
        return model.ℓπ(z) * β
    end
    return logπ
end
```

Access to the current proposed parameter values is require, this should be a relatively simple getter function accessing the `state` of the sampler:

```julia
function get_θ(state::HMCState)
    return state.z.θ
end
```

Both of these parts should then be used in a function called `get_densities_and_θs` that returns the densities and parameters for the `k`th and `k+1`th chains, the interface is built in this way as the requirements for accessing to the two aforementioned components can reasonably change, some require state information, sampler information, model information etc., this allows for flexibility in implementation.

In this case, the implementation of `get_densities_and_θs` is relatively simple, in fact, this code represents the default "fallback" implementation of this function and so provided your sampler submits to this fairly standard set of arguments you do not need to implement this method:

```julia
function get_densities_and_θ(
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

Feel free to override this functionality though, this is necessary in `Turing` for example where the `sampler` and `VarInfo` are required to access the density and parameters:

```julia
function get_densities_and_θs(
    model::Model,
    sampler::Sampler{<:TemperedAlgorithm},
    states,
    k::Integer,
    Δ::Vector{T},
    Δ_state::Vector{<:Integer}
) where {T<:AbstractFloat}

    logπk = make_tempered_logπ(model, Δ[Δ_state[k]], sampler, get_vi(states[k][2]))
    logπkp1 = make_tempered_logπ(model, Δ[Δ_state[k + 1]], sampler, get_vi(states[k + 1][2]))
    
    θk = get_θ(states[k][2], sampler)
    θkp1 = get_θ(states[k + 1][2], sampler)
    
    return logπk, logπkp1, θk, θkp1
end


function make_tempered_logπ(model::Model, β::T, sampler::DynamicPPL.Sampler, varinfo_init::DynamicPPL.VarInfo) where {T<:AbstractFloat}
    
    function logπ(z)
        varinfo = DynamicPPL.VarInfo(varinfo_init, sampler, z)
        model(varinfo)
        return DynamicPPL.getlogp(varinfo) * β
    end

    return logπ
end

get_vi(state::Union{HMCState,GibbsState,EmceeState,SMCState}) = state.vi
get_vi(vi::DynamicPPL.VarInfo) = vi

get_θ(state, sampler::DynamicPPL.Sampler) = get_vi(state)[sampler]
```
