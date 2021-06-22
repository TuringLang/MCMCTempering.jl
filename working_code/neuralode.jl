using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC
using JLD, StatsPlots, Distributions

u0 = [2.0; 0.0]
datasize = 40
tspan = (0.0, 1)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
mean_ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
ode_data = mean_ode_data .+ 0.1 .* randn(size(mean_ode_data)..., 30)

####DEFINE THE NEURAL ODE#####
dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, relu),
                  FastDense(50, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
end
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

function l(θ)
    lp = logpdf(MvNormal(zeros(length(θ) - 1), 1.0), θ[1:end-1])
    ll = sum(logpdf.(Normal.(ode_data, θ[end]), predict_neuralode(θ[1:end-1])))
    return lp + ll
end
function lp(θ)
    return logpdf(MvNormal(zeros(length(θ) - 1), 1.0), θ[1:end-1])
end
function ll(θ)
    return sum(logpdf.(Normal.(ode_data, θ[end]), predict_neuralode(θ[1:end-1])))
end

function dldθ(θ)
    x, lambda = Flux.Zygote.pullback(l,θ)
    grad = first(lambda(1))
    return x, grad
end
function dlpdθ(θ)
    x, lambda = Flux.Zygote.pullback(lp,θ)
    grad = first(lambda(1))
    return x, grad
end
function dlldθ(θ)
    x, lambda = Flux.Zygote.pullback(ll,θ)
    grad = first(lambda(1))
    return x, grad
end

init = [Float64.(prob_neuralode.p); 1.0]

opt = DiffEqFlux.sciml_train(x -> -l(x), init, ADAM(0.05), maxiters = 1500)
pmin = opt.minimizer;
metric  = DiagEuclideanMetric(length(pmin))
h = Hamiltonian(metric, l, dldθ)
integrator = Leapfrog(find_good_stepsize(h, pmin))
prop = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator, 10)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.5, prop.integrator))

samples, stats = sample(h, prop, pmin, 500, adaptor, 500; progress=true)

using MCMCTempering
tempered_samples = sample()