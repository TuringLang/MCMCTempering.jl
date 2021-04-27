using Turing
using StatsPlots

@model function gdemo(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

chn = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)
