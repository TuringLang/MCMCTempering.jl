using Plots
using Distributions
using AdvancedMH
include("src/temperature_scheduling.jl")

θᵣ = [-1., 1., 2., 1., 15., 2., 90., 1.5]
γs = [0.15, 0.25, 0.3, 0.3]

Δ = check_Δ([0, 0.01, 0.1, 0.25, 0.5, 1])

modelᵣ = MixtureModel(Distributions.Normal.(eachrow(reshape(θᵣ, (2,4)))...), γs)
xrange = -10:0.1:100
tempered_densities = pdf.(modelᵣ, xrange) .^ Δ'
norm_const = sum(tempered_densities[:,1])
for i in 2:length(Δ)
    tempered_densities[:,i] = (tempered_densities[:,i] ./ sum(tempered_densities[:,i])) .* norm_const
end
plot(xrange, tempered_densities, label = Δ')

data = rand(modelᵣ, 100)

insupport(θ) = all(reshape(θ, (2,4))[2,:] .≥ 0)
dist(θ) = MixtureModel(Distributions.Normal.(eachrow(reshape(θ, (2,4)))...), γs)
density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Construct a DensityModel.
model = DensityModel(density)

# Set up our sampler with a joint multivariate Normal proposal.
spl = RWMH(MvNormal(8,1))

chain = sample(model, spl, 100000; chain_type=Chains)

chain2 = SimulatedTempering(model, spl, Δ)