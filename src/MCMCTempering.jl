module MCMCTempering

import AbstractMCMC
import Distributions
import MCMCChains
import Random

include("tempered.jl")
include("ladders.jl")
include("stepping.jl")
include("model.jl")
include("swapping.jl")

export Tempered, TemperedSampler, plot_swaps

end
