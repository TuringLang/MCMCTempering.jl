"""
    struct TemperedSampler{A} <: AbstractSampler
        internal_sampler :: A
        Δ                :: Vector{<:AbstractFloat}
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
    Δ                :: Vector{<:AbstractFloat}
    Δ_init           :: Vector{<:Integer}
    N_swap           :: Integer
    swap_strategy    :: Symbol
end


function Tempered(
    internal_sampler,
    Δ::Vector{<:AbstractFloat};
    swap_strategy::Symbol = :standard,
    kwargs...
)
    return Tempered(internal_sampler, check_Δ(Δ), swap_strategy; kwargs...)
end
function Tempered(
    internal_sampler,
    Nt::Integer;
    swap_strategy::Symbol = :standard,
    kwargs...
)
    return Tempered(internal_sampler, generate_Δ(Nt, swap_strategy), swap_strategy; kwargs...)
end

"""
    Tempered

# Arguments
- `internal_sampler` is an algorithm or sampler object to be used for underlying sampling and to apply tempering to
- The temperature schedule can be defined either explicitly or just as an integer number of temperatures, i.e. as:
    - `Δ` containing a sequence of 'inverse temperatures' {β₀, ..., βₙ} where 0 ≤ βₙ < ... < β₁ < β₀ = 1
        OR
    - `Nt`, an integer specifying the number of inverse temperatures to include in a generated `Δ`

# Additional arguments
- `N_swap` steps are carried out between each tempering swap step attempt
- `swap_strategy` is the way in which temperature swaps are made, one of:
   `:standard` as in original proposed algorithm, a single randomly picked swap is proposed
   `:nonrev` alternate even/odd swaps as in Syed, Bouchard-Côté, Deligiannidis, Doucet, arXiv:1905.02939 such that a reverse swap cannot be made in immediate succession
   `:randperm` generates a permutation in order to swap in a random order
- `Δ_init` is a list containing the sequence of `1:length(Δ)` and determines the starting temperature of each chain
- TODO `swap_ar_target` defaults to 0.234 per REFERENCE
- TODO `store_swaps` is a flag determining whether to store the state of the chain after each swap move or not
"""
function Tempered(
    internal_sampler,
    Δ::Vector{<:AbstractFloat},
    swap_strategy::Symbol;
    Δ_init = collect(1:length(Δ)),
    N_swap::Integer = 1,
    kwargs...
)
    length(Δ) > 1 || error("More than one inverse temperatures must be provided.")
    N_swap >= 1 || error("This must be a positive integer.")
    return TemperedSampler(alg, Δ, Δ_init, N_swap, swap_strategy)
end
