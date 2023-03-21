@testset "AbstractMCMC" begin
    rng = Random.default_rng()
    model = DistributionLogDensity(MvNormal(Ones(2), I))
    logdensity_model = LogDensityModel(model)

    spl = RWMH(MvNormal(Zeros(dimension(model)), I))
    @test spl isa AbstractMCMC.AbstractSampler

    @testset "CompositionSampler(.., saveall=$(saveall))" for saveall in (true, false, Val(true), Val(false))
        spl_composed = MCMCTempering.CompositionSampler(spl, spl, saveall)

        num_iters = 100
        # Taking two steps with `spl` should be equivalent to one step with `spl ∘ spl`.
        # Use the same initial state.
        state_initial = last(AbstractMCMC.step(Random.default_rng(), logdensity_model, spl))
        state_composed_initial = MCMCTempering.state_from(
            logdensity_model,
            last(AbstractMCMC.step(Random.default_rng(), logdensity_model, spl_composed)),
            state_initial,
        )

        @test state_composed_initial isa MCMCTempering.CompositionState

        # Take two steps with `spl`.
        rng = Random.MersenneTwister(42)
        state = deepcopy(state_initial)
        for _ = 1:num_iters
            transition, state = AbstractMCMC.step(rng, logdensity_model, spl, state)
            transition, state = AbstractMCMC.step(rng, logdensity_model, spl, state)
        end
        params, logp = MCMCTempering.getparams_and_logprob(logdensity_model, state)

        # Take one step with `spl ∘ spl`.
        rng = Random.MersenneTwister(42)
        state_composed = deepcopy(state_composed_initial)
        for _ = 1:num_iters
            transition, state_composed = AbstractMCMC.step(rng, logdensity_model, spl_composed, state_composed)

            # Make sure the state types stay consistent.
            if MCMCTempering.saveall(spl_composed)
                @test transition isa MCMCTempering.CompositionTransition
            end
            @test state_composed isa MCMCTempering.CompositionState
        end
        params_composed, logp_composed = MCMCTempering.getparams_and_logprob(logdensity_model, state_composed)

        # Check that the parameters and log probability are the same.
        @test params == params_composed
        @test logp == logp_composed

        # Make sure that `AbstractMCMC.sample` is good.
        chain_composed = sample(logdensity_model, spl_composed, 2; progress=false, chain_type=MCMCChains.Chains)
        chain = sample(
            logdensity_model, spl, MCMCTempering.saveall(spl_composed) ? 4 : 2;
            progress=false, chain_type=MCMCChains.Chains
        )

        # Should be the same length because the `SequentialTransitions` will be unflattened.
        @test chain_composed isa MCMCChains.Chains
        @test length(chain_composed) == length(chain)
    end

    @testset "RepeatedSampler(..., saveall=$(saveall))" for saveall in (true, false, Val(true), Val(false))
        spl_repeated = MCMCTempering.RepeatedSampler(spl, 2, saveall)
        @test spl_repeated isa MCMCTempering.RepeatedSampler

        # Taking two steps with `spl` should be equivalent to one step with `spl ∘ spl`.
        # Use the same initial state.
        state_initial = last(AbstractMCMC.step(Random.default_rng(), logdensity_model, spl))
        state_repeated_initial = MCMCTempering.state_from(
            logdensity_model,
            last(AbstractMCMC.step(Random.default_rng(), logdensity_model, spl_repeated)),
            state_initial,
        )

        num_iters = 100

        # Take two steps with `spl`.
        rng = Random.MersenneTwister(42)
        state = deepcopy(state_initial)
        for _ = 1:num_iters
            transition, state = AbstractMCMC.step(rng, logdensity_model, spl, state)
            transition, state = AbstractMCMC.step(rng, logdensity_model, spl, state)
        end
        params, logp = MCMCTempering.getparams_and_logprob(logdensity_model, state)

        # Take one step with `spl ∘ spl`.
        rng = Random.MersenneTwister(42)
        state_repeated = deepcopy(state_repeated_initial)
        for _ = 1:num_iters
            transition, state_repeated = AbstractMCMC.step(rng, logdensity_model, spl_repeated, state_repeated)

            # Make sure the state types stay consistent.
            if MCMCTempering.saveall(spl_repeated)
                @test transition isa MCMCTempering.SequentialTransitions
                @test state_repeated isa MCMCTempering.SequentialStates
            else
                @test state_repeated isa typeof(state_initial)
            end
        end
        params_repeated, logp_repeated = MCMCTempering.getparams_and_logprob(logdensity_model, state_repeated)

        # Check that the parameters and log probability are the same.
        @test params == params_repeated
        @test logp == logp_repeated

        # Make sure that `AbstractMCMC.sample` is good.
        chain_repeated = sample(logdensity_model, spl_repeated, 2; progress=false, chain_type=MCMCChains.Chains)
        chain = sample(
            logdensity_model, spl, MCMCTempering.saveall(spl_repeated) ? 4 : 2;
            progress=false, chain_type=MCMCChains.Chains
        )

        # Should be the same length because the `SequentialTransitions` will be unflattened.
        @test length(chain_repeated) == length(chain)
    end

    @testset "MultiSampler" begin
        spl_multi = spl × spl
        @testset "$model_multi" for model_multi in [
            MCMCTempering.MultiModel((logdensity_model, logdensity_model)),  # tuple
            MCMCTempering.MultiModel([logdensity_model, logdensity_model]),  # vector
            MCMCTempering.MultiModel((m for m in [logdensity_model, logdensity_model]))  # iterator
            ]

            @test spl_multi isa MCMCTempering.MultiSampler

            num_iters = 100
            # Use the same initial state.
            states_initial = map(model_multi.models) do model
                last(AbstractMCMC.step(Random.default_rng(), model, spl))
            end
            states_multi_initial = MCMCTempering.state_from(
                model_multi,
                last(AbstractMCMC.step(Random.default_rng(), model_multi, spl_multi)),
                MCMCTempering.MultipleStates(states_initial),
            )

            params_and_logp_initial = map(
                Base.Fix1(MCMCTempering.getparams_and_logprob, logdensity_model),
                states_initial
            )
            params_multi_initial, logp_multi_initial = MCMCTempering.getparams_and_logprob(
                model_multi, states_multi_initial
            )
            @test map(first, params_and_logp_initial) == params_multi_initial
            @test map(last, params_and_logp_initial) == logp_multi_initial

            # Taking a step with `spl_multi` on `multimodel` should be equivalent
            # to stepping with the component samplers on the component models.
            rng = Random.MersenneTwister(42)
            rng_multi = Random.MersenneTwister(42)
            states = deepcopy(states_initial)
            state_multi = deepcopy(states_multi_initial)
            for _ = 1:num_iters
                state_multi = last(AbstractMCMC.step(rng_multi, model_multi, spl_multi, state_multi))
                states = map(model_multi.models, states) do model, state
                    last(AbstractMCMC.step(rng, model, spl, state))
                end
            end
            params_and_logp = map(Base.Fix1(MCMCTempering.getparams_and_logprob, logdensity_model), states)
            params_multi, logp_multi = MCMCTempering.getparams_and_logprob(model_multi, state_multi)

            @test map(first, params_and_logp) == params_multi
            @test map(last, params_and_logp) == logp_multi
        end
    end

    @testset "SwapSampler" begin
        # SwapSampler without tempering (i.e. in a composition and using `MultiModel`, etc.)
        init_params = [[5.0], [5.0]]
        mdl1 = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), DistributionLogDensity(Normal(4.9999, 1)))
        mdl2 = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), DistributionLogDensity(Normal(5.0001, 1)))
        spl1 = RWMH(MvNormal(Zeros(dimension(mdl1)), I))
        spl2 = let σ² = 1e-2
            MALA(∇ -> MvNormal(σ² * ∇, 2σ² * I))
        end
        swapspl = MCMCTempering.SwapSampler()
        spl_full = (spl1 × spl2) ∘ swapspl
        product_model = LogDensityModel(mdl1) × LogDensityModel(mdl2)
        # Sample!
        multisamples = sample(
            product_model, spl_full, 1000;
            init_params=init_params,
            progress=false,
            discard_initial=100,  # a bit of burn-in
        )
        # Extract the transitions corresponding to each of the models.
        model_transitions = mapreduce(hcat, multisamples) do t
            # Sort the transitions so they're in the order of the chains.
            x = MCMCTempering.sort_by_chain(
                MCMCTempering.ProcessOrder(),
                MCMCTempering.inner_transition(t),
                MCMCTempering.outer_transition(t).transitions
            )
            return [x...]
        end
        # Make sure we actually got some swaps going and we were using different types of states
        # for both models.
        @test length(unique(typeof, model_transitions[1, :])) ≥ 1
        @test length(unique(typeof, model_transitions[2, :])) ≥ 1

        # Check that means are roughly okay.
        model_params = map(first ∘ MCMCTempering.getparams, model_transitions)
        @test vec(mean(model_params; dims=2)) ≈ [5.0, 5.0] atol=0.2
    end
end
