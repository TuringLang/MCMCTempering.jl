
###
# AdvancedMH interface
###

function ParallelTempering(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer,
    Δ::Array{Float64,1};
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, sampler, N; Δ=Δ, kwargs...)
end

function ParallelTempering(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer,
    Nt::Integer;
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, sampler, N; Nt=Nt, gen_Δ=true, kwargs...)
end

function ParallelTempering(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer,
    Δ::Array{Float64,1},
    parallel::AbstractMCMC.AbstractMCMCParallel;
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, sampler, N, parallel; Δ=Δ, kwargs...)
end

function ParallelTempering(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer,
    Nt::Integer,
    parallel::AbstractMCMC.AbstractMCMCParallel;
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, sampler, N, parallel; Nt=Nt, gen_Δ=true, kwargs...)
end

###
# Turing interface
###

function ParallelTempering(
    model::AbstractMCMC.AbstractModel,
    alg::Turing.InferenceAlgorithm,
    N::Integer,
    Δ::Array{Float64,1};
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, alg, N, Δ; kwargs...)
end

function ParallelTempering(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    alg::Turing.InferenceAlgorithm,
    N::Integer,
    Δ::Array{Float64,1};
    kwargs...
)
    return ParallelTempering(rng, model, DynamicPPL.Sampler(alg, model), N; Δ=Δ, kwargs...)
end

function ParallelTempering(
    model::AbstractMCMC.AbstractModel,
    alg::Turing.InferenceAlgorithm,
    N::Integer,
    Nt::Integer;
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, alg, N, Nt; kwargs...)
end

function ParallelTempering(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    alg::Turing.InferenceAlgorithm,
    N::Integer,
    Nt::Integer;
    kwargs...
)
    return ParallelTempering(rng, model, DynamicPPL.Sampler(alg, model), N; Nt=Nt, gen_Δ=true, kwargs...)
end


function ParallelTempering(
    model::AbstractMCMC.AbstractModel,
    alg::Turing.InferenceAlgorithm,
    N::Integer,
    Δ::Array{Float64,1},
    parallel::AbstractMCMC.AbstractMCMCParallel;
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, alg, N, Δ, parallel; kwargs...)
end

function ParallelTempering(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    alg::Turing.InferenceAlgorithm,
    N::Integer,
    Δ::Array{Float64,1},
    parallel::AbstractMCMC.AbstractMCMCParallel;
    kwargs...
)
    return ParallelTempering(rng, model, DynamicPPL.Sampler(alg, model), N, parallel; Δ=Δ, kwargs...)
end

function ParallelTempering(
    model::AbstractMCMC.AbstractModel,
    alg::Turing.InferenceAlgorithm,
    N::Integer,
    Nt::Integer,
    parallel::AbstractMCMC.AbstractMCMCParallel;
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, alg, N, Nt, parallel; kwargs...)
end

function ParallelTempering(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    alg::Turing.InferenceAlgorithm,
    N::Integer,
    Nt::Integer,
    parallel::AbstractMCMC.AbstractMCMCParallel;
    kwargs...
)
    return ParallelTempering(rng, model, DynamicPPL.Sampler(alg, model), N, parallel; Nt=Nt, gen_Δ=true, kwargs...)
end


"""
    ParallelTempering

Samples `length(Δ)` parallel chains, each with `N` samples from `model` via parallel tempering using the `sampler` and temperature schedule `Δ`

# Arguments
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `N` samples (per chain) will be returned
- The temperature schedule can be defined either explicitly or just as an integer number of temperatures:
    - `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
        OR
    - `Nt` is an integer specifying the number of inverse temperatures to generate and run with

# Additional arguments
- `swap_strategy` is the way in which temperature swaps are made, one of:
   `:standard` as in original proposed algorithm, a single randomly picked swap is proposed
   `:nonrev` alternate even/odd swaps as in Syed, Bouchard-Côté, Deligiannidis, Doucet, arXiv:1905.02939 such that a reverse swap cannot be made in immediate succession
   `:randperm` generates a permutation in order to swap in a random order
- `swap_ar_target` defaults to 0.234 per REFERENCE
- `n_bs` steps are carried out between each tempering swap step attempt
- `n_warmup` is the number of initial samples to 'throw away'
- `thinning` defines how to thin the chains, every `thinning`th sample is stored; this defaults to 1 i.e. no thinning
- `store_swaps` is a flag determining whether to store the state of the chain after each swap move or not
- `progress` controls whether to show the progress meter or not
- `T₀` are a vector of the starting temperatures
- `chain_type` determines the output format, pick from `Any`, `Chains` or `StructArray`

# Outputs
- A list of chains, one for each temperature in Δ
- A list containing the temperature histories of each chain
- A list containing the indices relative to Δ of the temperature histories of each chain
"""
function ParallelTempering(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler,
    N;
    Δ::Array{Float64,1} = [1.0],
    Nt::Integer = 1,
    gen_Δ::Bool = false,
    swap_strategy::Symbol = :standard,
    swap_ar_target::Float64 = 0.234,
    n_bs::Integer = 1,
    n_warmup::Integer = 0,
    progress::Bool = true,
    T₀::Array{Int64,1} = collect(1:length(Δ)),
    chain_type = Any,
    kwargs...
)
    
    N > 0 || error("The number of samples to output must be ≥ 1")
    n_bs > 0 || error("The number of steps per iteration must be ≥ 1")

    n_warmup_iters = Int(ceil(n_warmup / (n_bs + store_swaps)))
    n_iters = Int(ceil((n_samples * thinning) / (n_bs + store_swaps)))

    if gen_Δ  # Need to generate a Δ given a number of temperatures
        Δ = generate_Δ(Nt, swap_strategy, swap_ar_target)
    else  # Δ has been passed manually, check it is valid
        Δ = check_Δ(Δ)
    end

    # Ts maintains the temperature ordering across the parallel chains
    Ts = T₀

    tempered_densities = get_tempered_densities(model, Δ, sampler)
    models = setup_models(model, Δ, sampler)

    interval = 1:length(Δ)
    rngs = [deepcopy(rng) for _ ∈ interval]
    samplers = [deepcopy(sampler) for _ ∈ interval]

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    AbstractMCMC.@ifwithprogresslogger progress name="Sampling" begin

        # initialise the parallel chains
        t, p_states, p_samples, p_chains, p_temperatures, p_temperature_indices = parallel_init_step(rng, model, sampler, Δ, Ts, Ntotal; kwargs...)

        for i ∈ 1:n_iters

            k = rand(Distributions.Categorical(length(Δ) - 1)) # Pick randomly from 1, 2, ..., k-1
            A = swap_acceptance_pt(model, p_samples[k], p_samples[k + 1], Δ[Ts[k]], Δ[Ts[k + 1]])

            U = rand(Distributions.Uniform(0, 1))
            # If the proposed temperature swap is accepted according to A and U, swap the temperatures for future steps
            if U ≤ A
                temp = Ts[k]
                Ts[k] = Ts[k + 1]
                Ts[k + 1] = temp
            end
            # Do a step without sampling to record the change in temperature
            t, p_chains, p_temperatures, p_temperature_indices = parallel_step_without_sampling(model, sampler, Δ, Ts, Ntotal, p_samples, p_chains, p_temperatures, p_temperature_indices, t; kwargs...)
            t, p_states, p_samples, p_chains, p_temperatures, p_temperature_indices = parallel_steps(rng, model, sampler, Δ, Ts, Ntotal, m, p_chains, p_temperatures, p_temperature_indices, p_states, t; kwargs...)
            
            progress && ProgressLogging.@logprogress (i / n_iters)

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
    n_iters::Integer = 1000,
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

    n_iters > 0 || error("The number of algorithm iterations must be ≥ 1")
    m > 0 || error("The number of proposals per iteration must be ≥ 1")
    Ntotal = n_iters * m

    if gen_Δ
        Δ = generate_Δ(Nt, swap_strategy, swap_ar_target)
    else  # Δ has been passed manually, check it is valid
        Δ = check_Δ(Δ)
    end

    @show Δ

    # Ts records the temperature ordering across the parallel chains
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

        Threads.@threads for i in 1:nchains
            # Obtain the ID of the current thread.
            id = Threads.threadid()

            # Seed the thread-specific random number generator with the pre-made seed.
            subrng = rngs[id]
            Random.seed!(subrng, seeds[i])

            t, p_states[i], p_samples[i], p_chains[i], p_temperatures[i], p_temperature_indices[i] = init_step(subrng, models[id], samplers[id], Δ[Ts[j]], Ts[j], Ntotal; kwargs...)
        end

        for i ∈ 1:n_iters

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

                # Do a step without sampling to record the change in temperature
                t, p_chains[j], p_temperatures[j], p_temperature_indices[j] = step_without_sampling(models[id], samplers[id], Δ[Ts[j]], Ts[j], Ntotal, p_samples[j], p_chains[j], p_temperatures[j], p_temperature_indices[j], t; kwargs...)
                t, p_states[j], p_samples[j], p_chains[j], p_temperatures[j], p_temperature_indices[j] = steps(rngs[id], models[id], samplers[id], Δ[Ts[j]], Ts[j], Ntotal, m, p_chains[j], p_temperatures[j], p_temperature_indices[j], p_states[j], t; kwargs...)
            end

            progress && ProgressLogging.@logprogress (i / n_iters)

        end
    end

    p_chains = reconstruct_chains(p_chains, p_temperature_indices, Δ)
    return [AbstractMCMC.bundle_samples(p_chains[i], model, sampler, p_states[i], chain_type) for i ∈ 1:length(Δ)], p_temperatures, p_temperature_indices

end

