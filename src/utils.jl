"""
    make_tempered_logπ

Constructs the likelihood density function for a `model` weighted by `β`

# Arguments
- The `model` in question
- An inverse temperature `β` with which to weight the density

## Notes
- For sake of efficiency, the returned function is closed over an instance of `VarInfo`. This means that you *might* run into some weird behaviour if you call this method sequentially using different types; if that's the case, just generate a new one for each type using `make_`.
"""
function make_tempered_logπ(model::DynamicPPL.Model, sampler::DynamicPPL.Sampler, varinfo_init::DynamicPPL.VarInfo)

    function logπ(z)
        varinfo = DynamicPPL.VarInfo(varinfo_init, sampler, z)
        model(varinfo)
        return DynamicPPL.getlogp(varinfo)
    end

    return logπ
end


"""
    get_vi

Returns the `VarInfo` portion of the `k`th chain's state contained in `states`

# Arguments
- `states` is 2D array containing `length(Δ)` pairs of transition + state for each chain
- `k` is the index of a chain in `states`
"""
get_vi(states, k::Integer) = get_vi(states[k][2])
get_vi(state::DynamicPPL.AbstractState) = state.vi
get_vi(vi::DynamicPPL.VarInfo) = vi


"""
    get_θ

Uses the `sampler` to index the `VarInfo` of the `k`th chain and return the associated `θ` proposal

# Arguments
- `states` is 2D array containing `length(Δ)` pairs of transition + state for each chain
- `k` is the index of a chain in `states`
- `sampler` is used to index the `VarInfo` such that `θ` is returned
"""
get_θ(states, k::Integer, sampler::DynamicPPL.Sampler) = get_vi(states, k)[sampler]
# get_vi(states, k)[DynamicPPL.SampleFromPrior()]


"""
    get_trans

Returns the `Transition` portion of the `k`th chain's state contained in `states`

# Arguments
- `states` is 2D array containing `length(Δ)` pairs of transition + state for each chain
- `k` is the index of a chain in `states`
"""
get_trans(states, k::Integer) = states[k][1]
