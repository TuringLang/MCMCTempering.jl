"""
    mutable struct TemperedAlgorithm <: Turing.InferenceAlgorithm
        alg           :: Turing.InferenceAlgorithm
        Δ             :: Vector{<:AbstractFloat}
        Δ_init        :: Vector{<:Integer}
        N_swap        :: Integer
        swap_strategy :: Symbol
    end

A `TemperedAlgorithm` struct wraps an `InferenceAlgorithm` `alg` alongside:
- A temperature ladder `Δ` containing a list of inverse temperatures `β`s
- The initial state of the tempered chains `Δ_init` in terms of which `β` each chain should begin at
- The number of steps between each temperature swap attempt `N_swap`
- The `swap_strategy` defining how these swaps should be carried out
"""
mutable struct TemperedAlgorithm <: Turing.InferenceAlgorithm
    alg           :: Turing.InferenceAlgorithm
    Δ             :: Vector{<:AbstractFloat}
    Δ_init        :: Vector{<:Integer}
    N_swap        :: Integer
    swap_strategy :: Symbol
end


function Tempered(
    alg::Turing.InferenceAlgorithm,
    Δ::Vector{<:AbstractFloat};
    swap_strategy::Symbol = :standard,
    kwargs...
)
    return Tempered(alg, check_Δ(Δ), swap_strategy; kwargs...)
end
function Tempered(
    alg::Turing.InferenceAlgorithm,
    Nt::Integer;
    swap_strategy::Symbol = :standard,
    kwargs...
)
    return Tempered(alg, generate_Δ(Nt, swap_strategy), swap_strategy; kwargs...)
end

"""
    Tempered

# Arguments
- `alg` an `InferenceAlgorithm` to be used for underlying sampling and to apply tempering to
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
    alg::Turing.InferenceAlgorithm,
    Δ::Vector{<:AbstractFloat},
    swap_strategy::Symbol;
    Δ_init = collect(1:length(Δ)),
    N_swap::Integer = 1,
    kwargs...
)
    length(Δ) >= 1 || error("More than one inverse temperatures must be provided.")
    N_swap >= 1 || error("This must be a positive integer.")
    return TemperedAlgorithm(alg, Δ, Δ_init, N_swap, swap_strategy)
end





# # assume
# function DynamicPPL.tilde(rng, ctx::DefaultContext, sampler{<:TemperedModelSampler}, right, vn::VarName, inds, vi)
#     ctx = DynamicPPL.MiniBatchContext(
#         ctx,
#         sampler.β
#     )
#     return DynamicPPL.tilde(rng, ctx, sampler.sampler, right, vn, inds, vi)
# end
# function DynamicPPL.tilde(rng, ctx::LikelihoodContext, sampler{<:TemperedSampler}, right, vn::VarName, inds, vi)
#     if ctx.vars isa NamedTuple && haskey(ctx.vars, getsym(vn))
#         vi[vn] = vectorize(right, _getindex(getfield(ctx.vars, getsym(vn)), inds))
#         settrans!(vi, false, vn)
#     end
#     return sampler.β * _tilde(rng, sampler.sampler, NoDist(right), vn, vi)
# end
# function DynamicPPL.tilde(rng, ctx::MiniBatchContext, sampler{<:TemperedSampler}, right, left::VarName, inds, vi)
#     return tilde(rng, ctx.ctx, sampler.sampler, right, left, inds, vi)
# end


