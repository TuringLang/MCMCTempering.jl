include("swap_acceptance.jl")
include("temperature_scheduling.jl")
include("stepping.jl")

function ParallelTempering(
    model,
    sampler,
    Δ::Array{Float64,1};
    kwargs...
)
    return ParallelTempering(Random.GLOBAL_RNG, model, sampler, Δ; kwargs...)
end

"""
    ParallelTempering

Samples `length(Δ)` parallel chains, each with `iters * m` samples from `model` via parallel tempering using the `sampler` and temperature schedule `Δ`
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
- `iters` PT algorithm iterations will be carried out
- `m` updates are carried out between each swap attempt
- `progress` controls whether to show the progress meter or not
- `T₀` are a vector of the starting temperatures
- `chain_type` determines the output format, pick from `Any`, `Chains` or `StructArray`
"""
function ParallelTempering(
    rng::Random.AbstractRNG,
    model,
    sampler,
    Δ::Array{Float64,1};
    iters = 100,
    m = 50,
    progress = true,
    T₀ = collect(1:length(Δ)),
    chain_type = Any,
    kwargs...
)
    
    iters > 0 || error("The number of algorithm iterations must be ≥ 1")
    m > 0 || error("The number of proposals per iteration must be ≥ 1")
    Ntotal = iters * m

    Δ = check_Δ(Δ)
    Ts = T₀
    progress_id = UUIDs.uuid1(rng)

    AbstractMCMC.@ifwithprogresslogger progress parentid=progress_id name="Sampling" begin

        # init step
        t, p_states, p_samples, p_chains, p_temperatures = parallel_init_step(rng, model, sampler, Δ, Ts, Ntotal, progress, progress_id; kwargs...)

        for i in 1:iters
            k = rand(Distributions.Categorical(length(Δ) - 1)) # Pick randomly from 1, 2, ..., k-1
            A = swap_acceptance_pt(model, p_samples[k], p_samples[k + 1], Δ[Ts[k]], Δ[Ts[k + 1]])

            U = rand(Distributions.Uniform(0, 1))
            if U ≤ A
                temp = Ts[k]
                Ts[k] = Ts[k + 1]
                Ts[k + 1] = temp
            end
            t, p_chains, p_temperatures = parallel_step_without_sampling(model, sampler, Δ, Ts, Ntotal, p_samples, p_chains, p_temperatures, t, progress, progress_id; kwargs...)
            t, p_states, p_samples, p_chains, p_temperatures = parallel_steps(rng, model, sampler, Δ, Ts, Ntotal, m, p_chains, p_temperatures, p_states, t, progress, progress_id; kwargs...)

        end

    end
    return [AbstractMCMC.bundle_samples(p_chains[i], model, sampler, p_states[i], chain_type; kwargs...) for i in 1:length(Δ)], p_temperatures

end