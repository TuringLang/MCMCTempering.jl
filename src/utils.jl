"""
    roundtrips(transitions)

Return sequence of `(start_index, turnpoint_index, end_index)`-triples representing roundtrips.
"""
function roundtrips(transitions::AbstractVector{<:TemperedTransition})
    return roundtrips(map(Base.Fix2(getproperty, :swaptransition), transitions))
end
function roundtrips(transitions::AbstractVector{<:SwapTransition})
    result = Tuple{Int,Int,Int}[]
    start_index, turn_index = 1, nothing
    for (i, t) in enumerate(transitions)
        n = length(t.chain_to_process)
        if isnothing(turn_index)
            # Looking for the turn.
            if chain_to_process(t, 1) == n
                turn_index = i
            end
        else
            # Looking for the return/end.
            if chain_to_process(t, 1) == 1
                push!(result, (start_index, turn_index, i))
                # Reset.
                start_index = i
                turn_index = nothing
            end
        end
    end

    return result
end
