# MCMCTempering.jl

MCMCTempering.jl provides implementations of MCMC sampling algorithms such as simulated and parallel tempering, that are robust to multi-modal target distributions. These algorithms leverage temperature scheduling to flatten out the target distribution and allow sampling to move more freely around a target's complete state space to better explore its mass.
