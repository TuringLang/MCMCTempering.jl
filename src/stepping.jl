"""
    init_step

Initialise a chain by applying the appropriate tempering to the model, calling `AbstractMCMC.step`, and storing the results in a collection of transitions o.e

# Arguments
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `β` chosen inverse temperature for the step
- `Ntotal` is the total number of steps to be taken during execution

# Outputs
- `t` is set to 1, counting the steps taken in this chain
- `state` contains the chain's current state
- `sample` contains the most recent sample from the chain
- `chain` contains the full collection of samples to date
- `temperatures` contains the temperature history of the chain
"""
function init_step(
    rng::Random.AbstractRNG, 
    model::AbstractMCMC.AbstractModel, 
    sampler::AbstractMCMC.AbstractSampler, 
    β::Float64,
    T::Integer,
    Ntotal::Integer;
    kwargs...
)

    f(θ) = model.logdensity(θ) * β
    modelᵦ = AdvancedMH.DensityModel(f)

    sample, state = AbstractMCMC.step(rng, modelᵦ, sampler; kwargs...)
    chain = AbstractMCMC.samples(sample, model, sampler, Ntotal; kwargs...)
    chain = AbstractMCMC.save!!(chain, sample, 1, modelᵦ, sampler, Ntotal; kwargs...)

    return 1, state, sample, chain, [β], [T]

end


"""
    step_without_sampling

Store the sample, temperature and iterate `t`, but do not actually make a new proposal (this is used during temperature swap moves)

# Arguments
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `β` chosen inverse temperature for the step
- `Ntotal` is the total number of steps to be taken during execution
- `chain` stores all of the samples gathered so far 
- `temperatures` stores the temperatures of the chain at each step
- `t` maintains the index of the current step of the process

# Outputs
- `t` iterated to account for the step taken
- `chain` contains the full collection of samples to date
- `temperatures` contains the temperature history of the chain
"""
function step_without_sampling(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    β::Float64,
    T::Integer,
    Ntotal::Integer,
    sample,
    chain,
    temperatures::Array{Float64,1},
    temperature_indices::Array{Int64,1},
    t::Int64;
    kwargs...
)

    t += 1
    chain = AbstractMCMC.save!!(chain, sample, t, model, sampler, Ntotal; kwargs...)
    push!(temperatures, β)
    push!(temperature_indices, T)

    return t, chain, temperatures, temperature_indices

end

# TemperedMHSampler

# function AbstractMCMC.step()

#     edit model
#     call step

# end

"""
    steps

Take `m` steps in the passed `chain` with the passed temperature via `β`

# Arguments
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `β` chosen inverse temperature for the step
- `Ntotal` is the total number of steps to be taken during execution
- `m` updates are to be carried out during the execution of this function
- `chain` stores all of the samples gathered so far 
- `temperatures` stores the temperatures of the chain at each step
- `state` contains the state of the chain
- `t` maintains the index of the current step of the process

# Outputs
- `t` iterated to account for the step taken
- `state` contains the chain's current state
- `sample` contains the most recent sample from the chain
- `chain` contains the full collection of samples to date
- `temperatures` contains the temperature history of the chain
"""
function steps(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    β::Float64,
    T::Integer,
    Ntotal::Integer,
    m::Integer,
    chain,
    temperatures::Array{Float64,1},
    temperature_indices::Array{Int64,1},
    state,
    t::Integer;
    kwargs...
)

    f(θ) = model.logdensity(θ) * β
    modelᵦ = AdvancedMH.DensityModel(f)

    for i in 1:m
        t += 1
        sample, state = AbstractMCMC.step(rng, modelᵦ, sampler, state; β=β, kwargs...)
        chain = AbstractMCMC.save!!(chain, sample, t, model, sampler, Ntotal; kwargs...)
        push!(temperatures, β)
        push!(temperature_indices, T)
    end
    return t, state, chain[end], chain, temperatures, temperature_indices

end


"""
    parallel_init_step

Initialise a chain for each temperature in `Δ` by taking the first `init_step` of each

# Arguments
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
- `Ntotal` is the total number of steps to be taken during execution

# Outputs
- `t` is set to 1, counting the steps taken in this chain
- `p_states` contains each chain's current state
- `p_samples` contains the most recent sample from each chain
- `p_chains` stores all of the samples gathered so far in each chain
- `p_temperatures` contains the temperature history of each chain
"""
function parallel_init_step(
    rng::Random.AbstractRNG, 
    model::AbstractMCMC.AbstractModel, 
    sampler::AbstractMCMC.AbstractSampler, 
    Δ::Array{Float64,1},
    Ts::Array{Int64,1},
    Ntotal::Integer;
    kwargs...
)

    t, p_states, p_samples, p_chains, p_temperatures, p_temperature_indices = unzip(map(i -> init_step(rng, model, sampler, Δ[Ts[i]], Ts[i], Ntotal; kwargs...), Ts))
    return t[1], p_states, p_samples, p_chains, p_temperatures, p_temperature_indices

end


"""
    parallel_step_without_sampling

Store the sample, temperature and iterate `t` for each chain, but do not actually make a new proposal (used during temperature swap moves)

# Arguments
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
- `Ntotal` is the total number of steps to be taken during execution
- `p_chains` stores all of the samples gathered so far in each chain
- `p_temperatures` contains the temperature history of each chain
- `t` maintains the index of the current step of the process

# Outputs
- `t` iterated to account for the step taken
- `p_chains` stores all of the samples gathered so far in each chain
- `p_temperatures` contains the temperature history of each chain
"""
function parallel_step_without_sampling(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    Δ::Array{Float64,1},
    Ts::Array{Int64,1},
    Ntotal::Integer,
    p_samples,
    p_chains,
    p_temperatures,
    p_temperature_indices,
    t::Integer;
    kwargs...
)
    t, p_chains, p_temperatures, p_temperature_indices = unzip(map(i -> step_without_sampling(model, sampler, Δ[Ts[i]], Ts[i], Ntotal, p_samples[i], p_chains[i], p_temperatures[i], p_temperature_indices[i], t; kwargs...), 1:length(Δ)))
    return t[1], p_chains, p_temperatures, p_temperature_indices

end


"""
    parallel_steps

Take `m` steps in each `chain` of `p_chains` using temperatures `Δ` ordered by the current schedule indices `Ts`

# Arguments
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
- `Ts` contains the current temperature indices, implicitly recording any previous temperature swap moves
- `Ntotal` is the total number of steps to be taken during execution
- `m` updates per chain are to be carried out during the execution of this function
- `p_chains` stores all of the samples gathered so far in each chain
- `p_temperatures` contains the temperature history of each chain
- `p_states` contains each chain's current state
- `t` maintains the index of the current step of the process

# Outputs
- `t` iterated to account for the step taken
- `p_states` contains each chain's current state
- `p_samples` contains the most recent sample from each chain
- `p_chains` stores all of the samples gathered so far in each chain
- `p_temperatures` contains the temperature history of each chain
"""
function parallel_steps(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    Δ::Array{Float64,1},
    Ts::Array{Int64,1},
    Ntotal::Integer,
    m::Integer,
    p_chains,
    p_temperatures,
    p_temperature_indices,
    p_states,
    t::Integer;
    kwargs...
)

    t, p_states, parallel_samples, p_chains, p_temperatures, p_temperature_indices = unzip(map(i -> steps(rng, model, sampler, Δ[Ts[i]], Ts[i], Ntotal, m, p_chains[i], p_temperatures[i], p_temperature_indices[i], p_states[i], t; kwargs...), 1:length(Δ)))
    return t[1], p_states, parallel_samples, p_chains, p_temperatures, p_temperature_indices

end
