
function unzip(out)
    collect(zip(out...))
end

"""
    step

# Arguments
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `β` chosen inverse temperature for the step
- `Ntotal` is the total number of steps to be taken during execution
"""
function init_step(
    rng::Random.AbstractRNG, 
    model, 
    sampler, 
    β,
    Ntotal,
    progress,
    progress_id; 
    kwargs...
)

    f(θ) = model.logdensity(θ) * β
    modelᵦ = AdvancedMH.DensityModel(f)

    sample, state = AbstractMCMC.step(rng, modelᵦ, sampler; kwargs...)
    chain = AbstractMCMC.samples(sample, model, sampler, Ntotal; kwargs...)
    chain = AbstractMCMC.save!!(chain, sample, 1, modelᵦ, sampler, Ntotal; kwargs...)
    # progress && ProgressLogging.@logprogress _id=progress_id t / Ntotal

    return 1, state, sample, chain, [β]

end


"""
    step_without_sampling

Store the sample, temperature and iterate `t`, but do not actually make a new proposal (used during temperature swap moves)
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `β` chosen inverse temperature for the step
- `Ntotal` is the total number of steps to be taken during execution
- `chain` stores all of the samples gathered so far 
- `temperatures` stores the temperatures of the chain at each step
- `t` counts the step of the process
"""
function step_without_sampling(
    model,
    sampler,
    β,
    Ntotal,
    sample,
    chain,
    temperatures,
    t,
    progress,
    progress_id;
    kwargs...
)
    t += 1
    chain = AbstractMCMC.save!!(chain, sample, t, model, sampler, Ntotal; kwargs...)
    push!(temperatures, β)
    # progress && ProgressLogging.@logprogress _id=progress_id t / Ntotal

    return t, chain, temperatures

end


"""
    steps

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
- `t` counts the step of the process
"""
function steps(
    rng::Random.AbstractRNG,
    model,
    sampler,
    β,
    Ntotal,
    m,
    chain,
    temperatures,
    state,
    t,
    progress,
    progress_id;
    kwargs...
)

    f(θ) = model.logdensity(θ) * β
    modelᵦ = AdvancedMH.DensityModel(f)
    for i in 1:m
        t += 1
        sample, state = AbstractMCMC.step(rng, modelᵦ, sampler, state; kwargs...)
        chain = AbstractMCMC.save!!(chain, sample, t, model, sampler, Ntotal; kwargs...)
        push!(temperatures, β)
        # progress && ProgressLogging.@logprogress _id=progress_id t / Ntotal
    end
    return t, state, chain[end], chain, temperatures

end


"""
    parallel_init_step

Initialise a chain for each temperature in `Δ` by taking the first step in each
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
- `Ntotal` is the total number of steps to be taken during execution
"""
function parallel_init_step(
    rng::Random.AbstractRNG, 
    model, 
    sampler, 
    Δ,
    Ts,
    Ntotal,
    progress,
    progress_id;
    kwargs...
)
    # p_states = Array{Any}(length(Δ))
    # p_samples = Array{Any}(length(Δ))
    # p_chains = [Any for i in 1:length(Δ)]
    # p_temperatures = [Any for i in 1:length(Δ)]
    Ts = collect(1:length(Δ))
    t, p_states, p_samples, p_chains, p_temperatures = unzip(map(i -> init_step(rng, model, sampler, Δ[Ts[i]], Ntotal, progress, progress_id; kwargs...), Ts))
    # for i in Ts
    #     t, p_states[i], p_samples[i], p_chains[i], p_temperatures[i] = init_step(rng, model, sampler, Δ[Ts[i]], Ntotal, progress, progress_id; kwargs...)
    # end
    return t[1], p_states, p_samples, p_chains, p_temperatures, Ts

end


"""
    parallel_step_without_sampling

Store the sample, temperature and iterate `t`, but do not actually make a new proposal (used during temperature swap moves)
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
- `Ntotal` is the total number of steps to be taken during execution
- `p_chains` stores all of the samples gathered so far 
- `p_temperatures` stores the temperatures of the chain at each step
- `t` counts the step of the process
"""
function parallel_step_without_sampling(
    model,
    sampler,
    Δ,
    Ts,
    Ntotal,
    p_samples,
    p_chains,
    p_temperatures,
    t,
    progress,
    progress_id;
    kwargs...
)
    t, p_chains, p_temperatures = unzip(map(i -> step_without_sampling(model, sampler, Δ[Ts[i]], Ntotal, p_samples[i], p_chains[i], p_temperatures[i], t, progress, progress_id; kwargs...), 1:length(Δ)))
    return t[1], p_chains, p_temperatures

end


"""
    parallel_steps

Store the sample, temperature and iterate `t`, but do not actually make a new proposal (used during temperature swap moves)
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
- `Ntotal` is the total number of steps to be taken during execution
- `m` updates are to be carried out during the execution of this function
- `p_chains` stores all of the samples gathered so far 
- `p_temperatures` stores the temperatures of the chain at each step
- `p_states` contains the state of the chain
- `t` counts the step of the process
"""
function parallel_steps(
    rng,
    model,
    sampler,
    Δ,
    Ts,
    Ntotal,
    m,
    p_chains,
    p_temperatures,
    p_states,
    t,
    progress,
    progress_id;
    kwargs...
)

    t, p_states, parallel_samples, p_chains, p_temperatures = unzip(map(i -> steps(rng, model, sampler, Δ[Ts[i]], Ntotal, m, p_chains[i], p_temperatures[i], p_states[i], t, progress, progress_id; kwargs...), 1:length(Δ)))
    return t[1], p_states, parallel_samples, p_chains, p_temperatures

end
