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
        state_composed_initial = MCMCTempering.state_from_state(
            logdensity_model,
            state_initial,
            last(AbstractMCMC.step(Random.default_rng(), logdensity_model, spl_composed))
        )

        if MCMCTempering.saveall(spl_composed)
            @test state_composed_initial isa MCMCTempering.SequentialStates
        else
            @test state_composed_initial isa MCMCTempering.CompositionState
        end

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
                @test transition isa MCMCTempering.SequentialTransitions
                @test state_composed isa MCMCTempering.SequentialStates
            else
                @test state_composed isa MCMCTempering.CompositionState
            end
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
        @test length(chain_composed) == length(chain)
    end

    @testset "RepeatedSampler(..., saveall=$(saveall))" for saveall in (true, false, Val(true), Val(false))
        spl_repeated = MCMCTempering.RepeatedSampler(spl, 2, saveall)
        @test spl_repeated isa MCMCTempering.RepeatedSampler

        # Taking two steps with `spl` should be equivalent to one step with `spl ∘ spl`.
        # Use the same initial state.
        state_initial = last(AbstractMCMC.step(Random.default_rng(), logdensity_model, spl))
        state_repeated_initial = MCMCTempering.state_from_state(
            logdensity_model,
            state_initial,
            last(AbstractMCMC.step(Random.default_rng(), logdensity_model, spl_repeated))
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
            states_multi_initial = MCMCTempering.state_from_state(
                model_multi,
                MCMCTempering.MultipleStates(states_initial),
                last(AbstractMCMC.step(Random.default_rng(), model_multi, spl_multi))
            )

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
end
