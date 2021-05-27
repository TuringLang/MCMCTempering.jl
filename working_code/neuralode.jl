using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 3.5)
tsteps = 0.0:0.1:3.5

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob_ode = ODEProblem(lotka_volterra!, u0, tspan, p)
mean_ode_data = Array(solve(prob_ode, Tsit5(), saveat = tsteps))
ode_data = mean_ode_data .+ 0.1 .* randn(size(mean_ode_data)..., 30)
dudt2 = FastChain(FastDense(2, 20, tanh), FastDense(20, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

l(θ) = -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ .* θ)
llikelihood(θ) = -sum(abs2, ode_data .- predict_neuralode(θ))
lprior(θ) = -sum(θ .* θ)

function dldθ(θ)
    x,lambda = Flux.Zygote.pullback(l,θ)
    grad = first(lambda(1))
    return x, grad
end
function dllikelihooddθ(θ)
    x,lambda = Flux.Zygote.pullback(llikelihood,θ)
    grad = first(lambda(1))
    return x, grad
end
function dlpriordθ(θ)
    x,lambda = Flux.Zygote.pullback(lprior,θ)
    grad = first(lambda(1))
    return x, grad
end

using MCMCTempering

metric = DenseEuclideanMetric(length(prob_neuralode.p))
model = DifferentiableDensityModel(Joint(lprior, llikelihood), Joint(dlpriordθ, dllikelihooddθ))
h = Hamiltonian(metric, model.ℓπ, model.∂ℓπ∂θ)
# h = Hamiltonian(metric, l, dldθ)
integrator = Leapfrog(find_good_stepsize(h, Float64.(prob_neuralode.p)))
prop = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator; max_depth = 5)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, prop.τ.integrator))
samples, stats = sample(h, prop, Float64.(prob_neuralode.p), 500, adaptor, 1000; progress=true)

sampler = HMCSampler(prop, metric, adaptor)
samplest = sample(model, Tempered(sampler, 4), 1000; discard_initial=500, save_state=true)
