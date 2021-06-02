using Turing, Distributions, DifferentialEquations 

# Import MCMCChain, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(14);
using MCMCTempering



function lotka_volterra(du,u,p,t)
    x, y = u
    α, β, γ, δ  = p
    du[1] = (α - β*y)x # dx =
    du[2] = (δ*x - γ)y # dy = 
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0,1.0]
prob1 = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
sol = solve(prob1,Tsit5())
plot(sol)

sol1 = solve(prob1,Tsit5(),saveat=0.1)
odedata = Array(sol1) + 0.8 * randn(size(Array(sol1)))
plot(sol1, alpha = 0.3, legend = false); scatter!(sol1.t, odedata')

Turing.setadbackend(:forwarddiff)

@model function fitlv(data, prob1)
    σ ~ InverseGamma(2, 3) # ~ is the tilde character
    α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ truncated(Normal(1.0,0.5),0,2)
    γ ~ truncated(Normal(3.0,0.5),1,4)
    δ ~ truncated(Normal(1.0,0.5),0,2)

    p = [α,β,γ,δ]
    prob = remake(prob1, p=p)
    predicted = solve(prob,Tsit5(),saveat=0.1)

    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end

model = fitlv(odedata, prob1)

# This next command runs 3 independent chains without using multithreading. 
chain1_mh = sample(model, MH(), MCMCThreads(), 10000, 4)
chain1_nuts = sample(model, NUTS(.65), 1000)
# chain1_mh_test = mapreduce(c -> sample(model, NUTS(), 1000), chainscat, 1:3)


chain2_mh = sample(model, Tempered(MH(), 2), MCMCThreads(), 10000, 4)
chain2_nuts = sample(model, Tempered(NUTS(.65), 2), 1000)

chain3_mh = sample(model, Tempered(MH(), 3), MCMCThreads(), 10000, 30)
chain3_nuts = sample(model, Tempered(NUTS(.65), 3), MCMCThreads(), 1000, 30)

chain4_mh = sample(model, Tempered(MH(), 4), MCMCThreads(), 10000, 30)
chain4_nuts = sample(model, Tempered(NUTS(.65), 4), MCMCThreads(), 1000, 30)


plot_swaps(chain2_mh)

plot(chain1_mh)
plot(chain2_mh)

interchain_stats(chain1_mh)
interchain_stats(chain2_mh)



# Pumas example

function theop_model_Depots1Central1(du, u, p, t)
    Depot, Central = u
    Ka, CL, Vc = p
    du[1] = -Ka * Depot # d Depot = 
    du[2] =  Ka * Depot - (CL / Vc) * Central # d Central =
end

u0 = [1.0, 1.0]
p = [2.0, 0.2, 0.8, 2.0]
prob = ODEProblem(theop_model_Depots1Central1,u0,(0.0, 10.0),p)
sol = solve(prob, Tsit5())
plot(sol)

@model function theopmodel_bayes(dv, SEX, WT)

    N = length(dv)
    
    θ ~ arraydist(truncated.(Normal.([2.0, 0.2, 0.8, 2.0], 1.0), 0.0, 10.0))

    ωKa ~ Gamma(1.0, 0.2)
    ωCL ~ Gamma(1.0, 0.2)
    ωVc ~ Gamma(1.0, 0.2)

    σ ~ Gamma(1.0, 0.5)

    ηKa ~ filldist(Normal(0.0, ωKa), N)
    ηCL ~ filldist(Normal(0.0, ωCL), N)
    ηVc ~ filldist(Normal(0.0, ωVc), N)

    for i in 1:N
        Ka = (SEX[i] == 1 ? θ[1] : θ[4]) * exp(ηKa[i])
        CL = θ[2]*(WT[i]/70) * exp(ηCL[i])
        Vc = θ[3] * exp(ηVc[i])

        p = [Ka, CL, Vc]
        prob = remake(prob1, p=p)
        predicted = solve(prob, Tsit5(), saveat=0.1)

        μ[i] = predicted[i,2] / Vc
        dv[i] .~ Normal.(μ, σ)
    end
    dv

end

using Pumas




# BayesNeuralODE example

using BayesNeuralODE

N = 1
prior_std = likelihood_std = 1.0
model = BNO.generate_turing_model(:spiral, N, prior_std, likelihood_std)

bno_chain_1 = sample(model, NUTS(.6), 100)

bno_chain_2 = sample(model, Tempered(NUTS(.6), 4), MCMCThreads(), 1000, 4)
