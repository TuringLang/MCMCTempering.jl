
mutable struct TemperedState
    states        :: Array{Any}
    Δ_state       :: Vector{<:Integer}
    step_counter  :: Integer
end


"""
For each `β` in `Δ`, carry out a step with the `TemperedModel` at that `β` inverse temperature,
resulting in a list of transitions and states, the first transition is then returned with the
rest of the information being stored in the state.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:TemperedAlgorithm};
    kwargs...
)
    states = [
        AbstractMCMC.step(
            rng,
            DynamicPPL.Model(model.name, TemperedEval(model, spl.alg.Δ[Δi]), model.args, model.defaults),
            DynamicPPL.Sampler(spl.alg.alg, model);
            kwargs...
        )
        for Δi in spl.alg.Δ_init        
    ]
    return states[1][1], TemperedState(states, spl.alg.Δ_init, 1)
end


function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:TemperedAlgorithm},
    ts;
    kwargs...
)
    if ts.step_counter == spl.alg.N_swap
        ts.Δ_state = swap_step(rng, model, spl, ts)
        ts.step_counter = 0
    else
        ts.states = [
            AbstractMCMC.step(
                rng,
                DynamicPPL.Model(model.name, TemperedEval(model, spl.alg.Δ[ts.Δ_state[i]]), model.args, model.defaults),
                DynamicPPL.Sampler(spl.alg.alg, model),
                ts.states[i][2];
                kwargs...
            )
            for i in 1:length(spl.alg.Δ)
        ]
        ts.step_counter += 1
    end
    return ts.states[1][1], ts
end



function swap_step(
    rng::Random.AbstractRNG,
    model::Turing.Model,
    spl::DynamicPPL.Sampler{<:TemperedAlgorithm},
    ts::TemperedState
)
    L = length(spl.alg.Δ) - 1
    sampler = DynamicPPL.Sampler(spl.alg.alg, model)

    if spl.alg.swap_strategy == :standard

        k = rand(rng, Distributions.Categorical(L)) # Pick randomly from 1, 2, ..., k-1
        Δ_state = swap_attempt(model, sampler, ts.states, k, spl.alg.Δ, ts.Δ_state)

    else
        
        levels = Vector{Int}(undef, L)

        if spl.alg.swap_strategy == :nonrev
            if ts.step_counter % (2 * spl.alg.N_swap) == 0
                levels = 1:2:L
            else
                levels = 2:2:L
            end
        elseif spl.alg.swap_strategy == :randperm
            randperm!(rng, levels)
        end

        for k in levels
            Δ_state = swap_attempt(model, sampler, ts.states, k, spl.alg.Δ, ts.Δ_state)
        end
    end

    return Δ_state
end
