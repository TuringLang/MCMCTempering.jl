module MCMCTempering

import AbstractMCMC
import Distributions
import Random

include("tempered.jl")
include("ladders.jl")
include("stepping.jl")
include("model.jl")
include("swapping.jl")

export Tempered, TemperedSampler, plot_swaps, make_tempered_model, get_densities_and_θs, make_tempered_logπ, get_θ

end
