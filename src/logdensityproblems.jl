implements_logdensity(x) = LogDensityProblems.capabilities(x) !== nothing

maybe_wrap_model(model) = implements_logdensity(model) ? AbstractMCMC.LogDensityModel(model) : model
maybe_wrap_model(model::AbstractMCMC.LogDensityModel) = model

"""
    logdensity(model, x)

Return the log-density of `model` at `x`.

!!! note
    By default this defers to `LogDensityProblems.logdensity` when `model` implements the `LogDensityProblems` interface.
    Otherwise, this method can be specifically implemented for `model`.
"""
function logdensity(model, x)
    if !implements_logdensity(model)
        error("`logdensity` is not implemented for `$(typeof(model))`; either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end
    return LogDensityProblems.logdensity(model, x)
end

logdensity(model::AbstractMCMC.LogDensityModel, x) = logdensity(model.logdensity, x)

"""
    logdensity_and_gradient(model, x)

Return the log-density and its gradient of `model` at `x`.

!!! note
    By default this defers to `LogDensityProblems.logdensity_and_gradient` when `model` implements the `LogDensityProblems` interface.
    Otherwise, this method can be specifically implemented for `model`.
"""
function logdensity_and_gradient(model, x)
    if !implements_logdensity(model)
        error("`logdensity_and_gradient` is not implemented for `$(typeof(model))`; either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end
    return LogDensityProblems.logdensity_and_gradient(model, x)
end
logdensity_and_gradient(model::AbstractMCMC.LogDensityModel, x) = logdensity_and_gradient(model.logdensity, x)


"""
    PowerTemperedLogDensityProblem

A power tempered log density function implementing the LogDensityProblem.jl interface.

This tempers the log density by raising it to the power of `beta`.

# Fields
$(FIELDS)
"""
struct PowerTemperedLogDensityProblem{L,T}
    "underlying log density; assumed to implement LogDensityProblems.jl interface"
    logdensity::L
    "tempering power"
    beta::T
end

LogDensityProblems.capabilities(::Type{<:PowerTemperedLogDensityProblem{L}}) where {L} = LogDensityProblems.capabilities(L)
LogDensityProblems.dimension(tf::PowerTemperedLogDensityProblem) = LogDensityProblems.dimension(tf.logdensity)
LogDensityProblems.logdensity(tf::PowerTemperedLogDensityProblem, x) = tf.beta * logdensity(tf.logdensity, x)
function LogDensityProblems.logdensity_and_gradient(tf::PowerTemperedLogDensityProblem, x)
    lp, ∇lp = logdensity_and_gradient(tf.logdensity, x)
    return tf.beta .* lp, tf.beta .* ∇lp
end

"""
    PathTemperedLogDensityProblem

A path tempered log density function implementing the LogDensityProblem.jl interface.

This tempers between a reference log density and the target log density.

# Fields
$(FIELDS)
"""
struct PathTemperedLogDensityProblem{L1,L2,T}
    "reference log density; assumed to implement both `rand` and`logdensity`"
    reference::L1
    "target log density; assumed to implement the LogDensityProblems.jl interface"
    target::L2
    "tempering power"
    beta::T
end

LogDensityProblems.capabilities(::Type{<:PathTemperedLogDensityProblem{L}}) where {L} = LogDensityProblems.capabilities(L)
LogDensityProblems.dimension(tf::PathTemperedLogDensityProblem) = LogDensityProblems.dimension(tf.target)
function LogDensityProblems.logdensity(tf::PathTemperedLogDensityProblem, x)
    lp_reference = logdensity(tf.reference, x)
    lp_target = logdensity(tf.target, x)
    return (1 - tf.beta) .* lp_reference + tf.beta .* lp_target
end
function LogDensityProblems.logdensity_and_gradient(tf::PathTemperedLogDensityProblem, x)
    lp_reference, ∇lp_reference = logdensity_and_gradient(tf.reference, x)
    lp_target, ∇lp_target = logdensity_and_gradient(tf.target, x)
    return (1 - tf.beta) .* lp_reference + tf.beta .* lp_target, (1 - tf.beta) .* ∇lp_reference + tf.beta .* ∇lp_target
end
