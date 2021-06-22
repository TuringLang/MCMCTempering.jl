
"""
When sample is called with the `save_state` kwarg set to `true`, the chain can be used to plot the tempering swaps that occurred during sampling
"""
function plot_swaps(chain)
    plot(chain.info.samplerstate.Δ_index_history)
end


function plot_ladders(chain)
    plot(chain.info.samplerstate.Δ_history)
end
