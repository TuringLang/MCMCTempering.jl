"""
    struct TemperedSampler{T} <: AbstractMCMC.AbstractSampler
        internal_sampler :: T
        Δ                :: Vector{<:Real}
        Δ_init           :: Vector{<:Integer}
        N_swap           :: Integer
        swap_strategy    :: Symbol
    end

A `TemperedSampler` struct wraps an `internal_sampler` (could just be an algorithm) alongside:
- A temperature ladder `Δ` containing a list of inverse temperatures `β`s
- The initial state of the tempered chains `Δ_init` in terms of which `β` each chain should begin at
- The number of steps between each temperature swap attempt `N_swap`
- The `swap_strategy` defining how these swaps should be carried out
"""
struct TemperedSampler{A} <: AbstractMCMC.AbstractSampler
    internal_sampler :: A
    Δ                :: Vector{<:Real}
    Δ_init           :: Vector{<:Integer}
    N_swap           :: Integer
    swap_strategy    :: Symbol
end


"""
    tempered(internal_sampler, Δ; <keyword arguments>)
    OR
    tempered(internal_sampler, Nt; <keyword arguments>)

# Arguments
- `internal_sampler` is an algorithm or sampler object to be used for underlying sampling and to apply tempering to
- The temperature schedule can be defined either explicitly or just as an integer number of temperatures, i.e. as:
    - `Δ::Vector{<:Real}` containing a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
        OR
    - `Nt::Integer`, specifying the number of inverse temperatures to include in a generated `Δ`
- `swap_strategy::Symbol` is the way in which temperature swaps are made, one of:
   `:standard` as in original proposed algorithm, a single randomly picked swap is proposed
   `:nonrev` alternate even/odd swaps as in Syed, Bouchard-Côté, Deligiannidis, Doucet, arXiv:1905.02939 such that a reverse swap cannot be made in immediate succession
   `:randperm` generates a permutation in order to swap in a random order
- `Δ_init::Vector{<:Integer}` is a list containing the sequence of `1:length(Δ)` and determines the starting temperature of each chain
- `N_swap::Integer` steps are carried out between each tempering swap step attempt
"""
function tempered(
    internal_sampler,
    Δ::Vector{<:Real};
    swap_strategy::Symbol = :standard,
    kwargs...
)
    return tempered(internal_sampler, check_Δ(Δ), swap_strategy; kwargs...)
end
function tempered(
    internal_sampler,
    Nt::Integer;
    swap_strategy::Symbol = :standard,
    kwargs...
)
    return tempered(internal_sampler, generate_Δ(Nt, swap_strategy), swap_strategy; kwargs...)
end
function tempered(
    internal_sampler,
    Δ::Vector{<:Real},
    swap_strategy::Symbol;
    Δ_init::Vector{<:Integer} = collect(1:length(Δ)),
    N_swap::Integer = 1,
    kwargs...
)
    length(Δ) > 1 || error("More than one inverse temperatures must be provided.")
    N_swap >= 1 || error("This must be a positive integer.")
    return TemperedSampler(internal_sampler, Δ, Δ_init, N_swap, swap_strategy)
end
