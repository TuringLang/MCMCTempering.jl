
function ParallelTempering(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    Δ::Array{Float64,1};
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, sampler, Δ; kwargs...)
end

function ParallelTempering(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    Δ::Array{Float64,1},
    parallel::AbstractMCMC.AbstractMCMCParallel;
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, sampler, Δ, parallel; kwargs...)
end

"""
    ParallelTempering

Samples `length(Δ)` parallel chains, each with `iters * m` samples from `model` via parallel tempering using the `sampler` and temperature schedule `Δ`

# Arguments
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1

# Additional arguments
- `swap_strategy::Symbol`: Swap strategy, one of:
   `:single` (default, single randomly picked swap)
   `:randperm` (swap in random order)
   `:sweep` (upward or downward sweep, picked at random)
   `:nonrev` (alternate even/odd sites as in Syed, Bouchard-Côté, Deligiannidis, Doucet, arXiv:1905.02939)
- `iters` PT algorithm iterations will be carried out
- `m` updates are carried out between each swap attempt
- `progress` controls whether to show the progress meter or not
- `T₀` are a vector of the starting temperatures
- `chain_type` determines the output format, pick from `Any`, `Chains` or `StructArray`

# Outputs
- A list of chains, one for each temperature in Δ
- A list containing the temperature histories of each chain
"""
function ParallelTempering(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    Δ::Array{Float64,1};
    swap_strategy::Symbol = :single,
    iters::Integer = 1000,
    m::Integer = 50,
    progress::Bool = true,
    T₀::Array{Int64,1} = collect(1:length(Δ)),
    chain_type = Any,
    kwargs...
)
    
    iters > 0 || error("The number of algorithm iterations must be ≥ 1")
    m > 0 || error("The number of proposals per iteration must be ≥ 1")
    Ntotal = iters * m

    # Ensure Δ is ∈ the correct form
    Δ = check_Δ(Δ)
    # Ts maintains the temperature ordering across the parallel chains
    Ts = T₀

    AbstractMCMC.@ifwithprogresslogger progress name="Sampling" begin

        # initialise the parallel chains
        t, p_states, p_samples, p_chains, p_temperatures, p_temperature_indices = parallel_init_step(rng, model, sampler, Δ, Ts, Ntotal; kwargs...)

        for i ∈ 1:iters

            k = rand(Distributions.Categorical(length(Δ) - 1)) # Pick randomly from 1, 2, ..., k-1
            A = swap_acceptance_pt(model, p_samples[k], p_samples[k + 1], Δ[Ts[k]], Δ[Ts[k + 1]])

            U = rand(Distributions.Uniform(0, 1))
            # If the proposed temperature swap is accepted according to A and U, swap the temperatures for future steps
            if U ≤ A
                temp = Ts[k]
                Ts[k] = Ts[k + 1]
                Ts[k + 1] = temp
            end
            # Do a step without sampling to record the change ∈ temperature
            t, p_chains, p_temperatures, p_temperature_indices = parallel_step_without_sampling(model, sampler, Δ, Ts, Ntotal, p_samples, p_chains, p_temperatures, p_temperature_indices, t; kwargs...)
            t, p_states, p_samples, p_chains, p_temperatures, p_temperature_indices = parallel_steps(rng, model, sampler, Δ, Ts, Ntotal, m, p_chains, p_temperatures, p_temperature_indices, p_states, t; kwargs...)
            
            progress && ProgressLogging.@logprogress (i / iters)

        end

    end
    p_chains = reconstruct_chains(p_chains, p_temperature_indices, Δ)
    return [AbstractMCMC.bundle_samples(p_chains[i], model, sampler, p_states[i], chain_type) for i ∈ 1:length(Δ)], p_temperatures, p_temperature_indices

end


function ParallelTempering(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    Δ::Array{Float64,1},
    ::AbstractMCMC.MCMCThreads;
    swap_strategy::Symbol = :single,
    iters::Integer = 1000,
    m::Integer = 50,
    progress::Bool = true,
    T₀::Array{Int64,1} = collect(1:length(Δ)),
    chain_type = Any,
    kwargs...
)

    # Check if actually multiple threads are used.
    if Threads.nthreads() == 1
        @warn "Only a single thread available: MCMC chains are not sampled in parallel"
    end

    iters > 0 || error("The number of algorithm iterations must be ≥ 1")
    m > 0 || error("The number of proposals per iteration must be ≥ 1")
    Ntotal = iters * m

    # Ensure Δ is ∈ the correct form
    Δ = check_Δ(Δ)
    # Ts maintains the temperature ordering across the parallel chains
    Ts = T₀

    nchains = length(Δ)
    interval = 1:min(nchains, Threads.nthreads())
    rngs = [deepcopy(rng) for _ ∈ interval]
    models = [deepcopy(model) for _ ∈ interval]
    samplers = [deepcopy(sampler) for _ ∈ interval]

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    AbstractMCMC.@ifwithprogresslogger progress name="Sampling ($(min(length(Δ), Threads.nthreads())) threads)" begin
        
        # initialise the parallel chains
        t, p_states, p_samples, p_chains, p_temperatures, p_temperature_indices = parallel_init_step(rng, model, sampler, Δ, Ts, Ntotal; kwargs...)

        Threads.@threads for j in 1:nchains
            # Obtain the ID of the current thread.
            id = Threads.threadid()

            # Seed the thread-specific random number generator with the pre-made seed.
            subrng = rngs[id]
            Random.seed!(subrng, seeds[j])

            t, p_states[j], p_samples[j], p_chains[j], p_temperatures[j], p_temperature_indices[j] = init_step(subrng, models[id], samplers[id], Δ[Ts[j]], Ts[j], Ntotal; kwargs...)
        end

        for i ∈ 1:iters

            k = rand(Distributions.Categorical(length(Δ) - 1)) # Pick randomly from 1, 2, ..., k-1
            A = swap_acceptance_pt(model, p_samples[k], p_samples[k + 1], Δ[Ts[k]], Δ[Ts[k + 1]])

            U = rand(Distributions.Uniform(0, 1))
            # If the proposed temperature swap is accepted according to A and U, swap the temperatures for future steps
            if U ≤ A
                temp = Ts[k]
                Ts[k] = Ts[k + 1]
                Ts[k + 1] = temp
            end

            Threads.@threads for j in 1:nchains
                # Obtain the ID of the current thread.
                id = Threads.threadid()

                # Do a step without sampling to record the change ∈ temperature
                t, p_chains[j], p_temperatures[j], p_temperature_indices[j] = step_without_sampling(models[id], samplers[id], Δ[Ts[j]], Ts[j], Ntotal, p_samples[j], p_chains[j], p_temperatures[j], p_temperature_indices[j], t; kwargs...)
                t, p_states[j], p_samples[j], p_chains[j], p_temperatures[j], p_temperature_indices[j] = steps(rngs[id], models[id], samplers[id], Δ[Ts[j]], Ts[j], Ntotal, m, p_chains[j], p_temperatures[j], p_temperature_indices[j], p_states[j], t; kwargs...)
            end

            progress && ProgressLogging.@logprogress (i / iters)

        end
    end

    p_chains = reconstruct_chains(p_chains, p_temperature_indices, Δ)
    return [AbstractMCMC.bundle_samples(p_chains[i], model, sampler, p_states[i], chain_type) for i ∈ 1:length(Δ)], p_temperatures, p_temperature_indices

end

