function SimulatedTempering(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    Δ::Array{Float64,1};
    kwargs...
)
    return SimulatedTempering(Random.GLOBAL_RNG, model, sampler, Δ; kwargs...)
end


"""
    SimulatedTempering

Samples `iters * m` samples from `model` via simulated tempering using the `sampler` and temperature schedule `Δ`

# Arguments
- `rng` random number generator provision
- `model` an AbstractModel implementation defining the density likelihood for sampling
- `sampler` a within-temperature proposal mechanism to update the Χ-marginal, from qᵦ(x, .)
- `Δ` contains a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
- `iters` ST algorithm iterations will be carried out
- `K` defines a temperate normalising function ensuring the target is correctly proportional in the AR
- `m` updates are carried out between each swap attempt
- `progress` controls whether to show the progress meter or not
- `T₀` is the starting temperature
- `chain_type` determines the output format, pick from `Any`, `Chains` or `StructArray`

# Outputs
- A chain of the specified output type
- A list containing the temperature history of the chain
"""
function SimulatedTempering(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    Δ::Array{Float64,1};
    iters::Integer = 2000,
    K = (f(β) = 1),
    m::Integer = 50,
    progress::Bool = true,
    T₀::Integer = 1,
    chain_type = Any,
    kwargs...
)

    iters > 0 || error("The number of algorithm iterations must be ≥ 1")
    m > 0 || error("The number of proposals per iteration must be ≥ 1")
    Ntotal = iters * m

    # Turn Δ into an 'AbstractTemperatureSchedule' o.e.? Could enforce format via check_Δ and clean up?
    Δ = check_Δ(Δ)
    # Ts maintains the temperature ordering across the parallel chains
    T = T₀
    progress_id = UUIDs.uuid1(rng)

    AbstractMCMC.@ifwithprogresslogger progress parentid=progress_id name="Sampling" begin

        # initialise the chain at the chosen starting temperature T₀
        t, state, sample, chain, temperatures = init_step(rng, model, sampler, Δ[T], Ntotal, progress, progress_id; kwargs...)

        for i in 1:iters
            w = rand(Distributions.DiscreteNonParametric([-1, 1], [0.5, 0.5]))
            T′ = max(min(T + w, length(Δ)), 1) # If move would result in invalid temp then dont change, via max and min here
            A = swap_acceptance_st(model, sample, Δ[T′], Δ[T], K)

            U = rand(Distributions.Uniform(0, 1))
            # Update temperature if the proposed temperature is accepted according to A and U
            if U ≤ A
                T = T′
            end
            # Do a step without sampling to record the change in temperature
            t, chain, temperatures = step_without_sampling(model, sampler, Δ[T], Ntotal, sample, chain, temperatures, t, progress, progress_id; kwargs...)
            t, state, sample, chain, temperatures = steps(rng, model, sampler, Δ[T], Ntotal, m, chain, temperatures, state, t, progress, progress_id; kwargs...)
            
        end

    end
    return AbstractMCMC.bundle_samples(chain, model, sampler, state, chain_type; kwargs...), temperatures

end
