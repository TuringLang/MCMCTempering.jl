
# struct TemperedModel <: AbstractPPL.AbstractProbabilisticProgram
#     model :: DynamicPPL.Model
#     β     :: AbstractFloat
# end

struct TemperedEval
    model :: DynamicPPL.Model
    β     :: AbstractFloat
end


function (te::TemperedEval)(
    rng,
    model,
    varinfo,
    sampler,
    context,
    args...
)
    context = DynamicPPL.MiniBatchContext(
        context,
        te.β
    )
    te.model.f(rng, model, varinfo, sampler, context, args...)
end

# """
#     (tm::TemperedModel)([rng, varinfo, sampler, context])
# """
# function (tm::TemperedModel)(
#     rng::Random.AbstractRNG,
#     varinfo::DynamicPPL.AbstractVarInfo = DynamicPPL.VarInfo(),
#     sampler::AbstractMCMC.AbstractSampler = DynamicPPL.SampleFromPrior(),
#     context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext(),
# )
#     context = DynamicPPL.MiniBatchContext(
#         context,
#         tm.β
#     )
#     tm.model(rng, varinfo, sampler, context)
# end

# function (tm::TemperedModel)(args...)
#     return tm(Random.GLOBAL_RNG, args...)
# end
# # without VarInfo
# function (tm::TemperedModel)(
#     rng::Random.AbstractRNG,
#     sampler::AbstractMCMC.AbstractSampler,
#     args...,
# )
#     return tm(rng, DynamicPPL.VarInfo(), sampler, args...)
# end
# # without VarInfo and without AbstractSampler
# function (tm::TemperedModel)(
#     rng::Random.AbstractRNG,
#     context::DynamicPPL.AbstractContext
# )
#     return tm(rng, DynamicPPL.VarInfo(), DynamicPPL.SampleFromPrior(), context)
# end
