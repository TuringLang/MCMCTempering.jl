using Plots
using Distributions
using AdvancedMH
using MCMCChains
using MCMCTempering
using BayesNeuralODE
import Bijectors, Turing, Flux, Optim, AdvancedVI

# Trivial Normal Example

θᵣ = [-1., 2.]
Δ = [0.1, 0.25, 0.5, 1]

modelᵣ = Distributions.Normal(θᵣ...)
xrange = -7:0.1:5
tempered_densities = pdf.(modelᵣ, xrange) .^ Δ'
norm_const = sum(tempered_densities[:,1])
for i in 2:length(Δ)
    tempered_densities[:,i] = (tempered_densities[:,i] ./ sum(tempered_densities[:,i])) .* norm_const
end
plot(xrange, tempered_densities, label = Δ')

data = rand(modelᵣ, 1000)

insupport(θ) = (θ[2] ≥ 0)
dist(θ) = Distributions.Normal(θ...)
density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Construct a DensityModel
model = DensityModel(density)

# Set up our sampler with a joint multivariate Normal proposal
spl = RWMH(MvNormal(2,1))

chain = sample(model, spl, 100000; chain_type=Chains)
# chain2 = SimulatedTempering(model, spl, Δ)
p_chains, p_temperatures, p_temperature_indices = ParallelTempering(model, spl, Δ, MCMCThreads(); iters=1000, m=50, chain_type=Chains)
# p_chains, p_temperatures, p_temperature_indices = ParallelTempering(model, spl, Δ; iters=1000, m=100, chain_type=Chains)



# Slightly less trivial 1D Gaussian mixture

θᵣ = [-3., 1., 3., 1.5, 15., 1., 90., 1.]
γs = [0.1, 0.4, 0.3, 0.3]
γs = γs / sum(γs)
N = convert(Integer, length(θᵣ) / 2)

Δ = [0.0025, 0.05, 1]

modelᵣ = MixtureModel(Distributions.Normal.(eachrow(reshape(θᵣ, (2,N)))...), γs)
xrange = -10:0.1:100
tempered_densities = pdf.(modelᵣ, xrange) .^ Δ'
norm_const = sum(tempered_densities[:,1])
for i in 2:length(Δ)
    tempered_densities[:,i] = (tempered_densities[:,i] ./ sum(tempered_densities[:,i])) .* norm_const
end
plot(xrange, tempered_densities, label = Δ')

data = rand(modelᵣ, 100)

insupport(θ) = all(reshape(θ, (2,N))[2,:] .≥ 0)
dist(θ) = MixtureModel(Distributions.Normal.(eachrow(reshape(θ, (2,N)))...), γs)
density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Construct a DensityModel
model = DensityModel(density)

# Set up our sampler with a joint multivariate Normal proposal
spl = RWMH(MvNormal(N*2,1))

p_chains, p_temperatures, p_temperature_indices = ParallelTempering(model, spl, Δ, MCMCThreads(); iters=1000, m=50, chain_type=Chains)
f(x) = 0.01302558 - (-0.1833562/-2.765317) * (1 - exp(2.765317*x))
st_chain, temperatures = SimulatedTempering(model, spl, Δ; iters=10000, m=50, K=f, chain_type=Chains)
chain = sample(model, spl, 500000; chain_type=Chains)
# chain2 = SimulatedTempering(model, spl, Δ)
# p_chains, p_temperatures, p_temperature_indices = ParallelTempering(model, spl, Δ; chain_type=Chains)


N = 1
prior_std = likelihood_std = 1.0
model = BNO.generate_turing_model(:spiral, N, prior_std, likelihood_std)

p_chains, p_temperatures, p_temperature_indices = ParallelTempering(model, spl, Δ, MCMCThreads(); chain_type=Chains)





# Slightly less trivial 2D Gaussian mixture

Δ = [0.001, 0.1, 1]

θ₁ = reduce(hcat, reshape([[i, j] for i in 0.5:1:1.5 for j in 0.5:1:1.5], 4))'
θ₂ = reduce(hcat, reshape([[i, j] for i in 0:2:2 for j in 0:2:2], 4))'
θ₃ = reduce(hcat, reshape([[i, j] for i in -2:2:4 for j in -2:2:4], 16))'

model₁ = MixtureModel(Distributions.MvNormal.(eachrow(θ₁), 0.1))
model₂ = MixtureModel(Distributions.MvNormal.(eachrow(θ₂), 0.1))
model₃ = MixtureModel(Distributions.MvNormal.(eachrow(θ₃), 0.1))

range = -3:0.1:5
plot(
    contour(range, range, (x, y) -> logpdf(model₁, [x, y]), title="Model 1 (Easy)"),
    contour(range, range, (x, y) -> logpdf(model₂, [x, y]), title="Model 2 (Further Modes)"),
    contour(range, range, (x, y) -> logpdf(model₃, [x, y]), title="Model 3 (More Modes)"),
    layout=(3,1),
    size=(800,1600)
)




# Multimodal target of dimension d.
function multimodalTarget(d::Int, m=[0.0 0.0; 2.0 0.0; 0.0 2.0; 2.0 2.0]', sigma2=0.1^2, sigman=sigma2)
    # The means of mixtures
    n_m = size(m,2)
    @assert d>=2 "Dimension should be >= 2"
    let m=m, n_m=size(m,2), d=d
        function log_p(x::Vector{Float64})
            l_dens = -0.5*(mapslices(sum, (m.-x[1:2]).^2, dims=1)/sigma2)
            if d>2
                l_dens .-= 0.5*mapslices(sum, x[3:d].^2, dims=1)/sigman
            end
            l_max = maximum(l_dens) # Prevent underflow by log-sum trick
            l_max + log(sum(exp.(l_dens.-l_max)))
        end
    end
end



using AdaptiveMCMC


n = 100_000; L = 3
standard_chain₁ = adaptive_rwm(zeros(2), multimodalTarget(2, θ₁'), n; thin=1)
pt_chain₁ = adaptive_rwm(zeros(2), multimodalTarget(2, θ₁'), n; L = L, thin=1)

standard_chain₂ = adaptive_rwm(zeros(2), multimodalTarget(2, θ₂'), n; thin=1)
pt_chain₂ = adaptive_rwm(zeros(2), multimodalTarget(2, θ₂'), n; L = L, thin=1)

standard_chain₃ = adaptive_rwm(zeros(2), multimodalTarget(2, θ₃'), n; thin=1)
pt_chain₃ = adaptive_rwm(zeros(2), multimodalTarget(2, θ₃'), n; L = L, thin=1)

std_chn₁ = Chains(standard_chain₁.X')
std_chn₂ = Chains(standard_chain₂.X')
std_chn₃ = Chains(standard_chain₃.X')
pt_chn₁ = Chains(pt_chain₁.X')
pt_chn₂ = Chains(pt_chain₂.X')
pt_chn₃ = Chains(pt_chain₃.X')

# Assuming you have 'Plots' installed:
using Plots
p = plot(
    contour(range, range, (x, y) -> logpdf(model₁, [x, y]), title="Model 1 (Easy)"),
    scatter(std_chn₁[:param_1], std_chn₁[:param_2], title="AdvancedMH RWMH", legend=:none),
    scatter(pt_chn₁[:param_1], pt_chn₁[:param_2], title="Parallel Tempering RWMH", legend=:none),
    contour(range, range, (x, y) -> logpdf(model₂, [x, y]), title="Model 2 (Further Modes)"),
    scatter(std_chn₂[:param_1], std_chn₂[:param_2], legend=:none),
    scatter(pt_chn₂[:param_1], pt_chn₂[:param_2], legend=:none),
    contour(range, range, (x, y) -> logpdf(model₃, [x, y]), title="Model 3 (More Modes)"),
    scatter(std_chn₃[:param_1], std_chn₃[:param_2], legend=:none),
    scatter(pt_chn₃[:param_1], pt_chn₃[:param_2], legend=:none),
    layout=(3,3),
    size=(1200,1200)
)
savefig(p, "scatter.png")

p = plot(std_chn₁)
savefig(p, "model1_std.png")
p = plot(pt_chn₁)
savefig(p, "model1_pt.png")
p = plot(std_chn₁[1000:2000])
savefig(p, "model1_std_trimmed.png")
p = plot(pt_chn₁[1000:2000])
savefig(p, "model1_pt_trimmed.png")

p = plot(std_chn₂)
savefig(p, "model2_std.png")
p = plot(pt_chn₂)
savefig(p, "model2_pt.png")
p = plot(std_chn₂[1000:2000])
savefig(p, "model2_std_trimmed.png")
p = plot(pt_chn₂[1000:2000])
savefig(p, "model2_pt_trimmed.png")

p = plot(std_chn₃)
savefig(p, "model3_std.png")
p = plot(pt_chn₃)
savefig(p, "model3_pt.png")
p = plot(std_chn₃[1000:2000])
savefig(p, "model3_std_trimmed.png")
p = plot(pt_chn₃[1000:2000])
savefig(p, "model3_pt_trimmed.png")
