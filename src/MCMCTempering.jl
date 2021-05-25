module MCMCTempering

import AbstractMCMC
import Distributions
import Random

include("tempered.jl")
include("ladders.jl")
include("stepping.jl")
include("model.jl")
include("swapping.jl")
include("plotting.jl")

export Tempered, TemperedSampler, plot_swaps, make_tempered_model, get_densities_and_θs, make_tempered_logπ, get_θ

function AbstractMCMC.bundle_samples(
    ts::Vector,
    model::AbstractMCMC.AbstractModel,
    sampler::TemperedSampler,
    state::TemperedState,
    chain_type::Type;
    kwargs...
)
    AbstractMCMC.bundle_samples(ts, model, sampler.internal_sampler, state, chain_type; kwargs...)
end

end
