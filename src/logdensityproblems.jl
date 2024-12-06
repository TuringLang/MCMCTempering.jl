"""
    DistributionLogDensityProblem

A wrapper around a `Distribution` implementing the LogDensityProblem.jl interface.

# Fields
$(FIELDS)
"""
struct DistributionLogDensityProblem{D<:Distributions.Distribution}
    "distribution"
    dist::D
end

LogDensityProblems.capabilities(::Type{<:DistributionLogDensityProblem}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(dp::DistributionLogDensityProblem) = length(dp.dist)
LogDensityProblems.logdensity(dp::DistributionLogDensityProblem, x) = logdensity(dp.dist, x)

"""
    TemperedLogDensityProblem

A tempered log density function implementing the LogDensityProblem.jl interface.

# Fields
$(FIELDS)
"""
struct TemperedLogDensityProblem{L,T}
    "underlying log density; assumed to implement LogDensityProblems.jl interface"
    logdensity::L
    beta::T
end

LogDensityProblems.capabilities(::Type{<:TemperedLogDensityProblem{L}}) where {L} = LogDensityProblems.capabilities(L)
LogDensityProblems.dimension(tf::TemperedLogDensityProblem) = LogDensityProblems.dimension(tf.logdensity)
LogDensityProblems.logdensity(tf::TemperedLogDensityProblem, x) = tf.beta * logdensity(tf.logdensity, x)
function LogDensityProblems.logdensity_and_gradient(tf::TemperedLogDensityProblem, x)
    y, ∇y = logdensity_and_gradient(tf.logdensity, x)
    return tf.beta * y, tf.beta * ∇y
end

"""
    PathTemperedLogDensityProblem

A path tempered log density function implementing the LogDensityProblem.jl interface.

# Fields
$(FIELDS)
"""
struct PathTemperedLogDensityProblem{L,D,T}
    "underlying log density; assumed to implement LogDensityProblems.jl interface"
    logdensity::L
    "reference"
    reference::D
    "(inverse) temperature"
    beta::T
end

# FIXME: capabilities should technically rely on both the logdensity and the reference.
LogDensityProblems.capabilities(::Type{<:PathTemperedLogDensityProblem{L}}) where {L} = LogDensityProblems.capabilities(L)
LogDensityProblems.dimension(tf::PathTemperedLogDensityProblem) = LogDensityProblems.dimension(tf.logdensity)
function LogDensityProblems.logdensity(tf::PathTemperedLogDensityProblem, x)
    return tf.beta * logdensity(tf.logdensity, x) + (1 - tf.beta) * logdensity(tf.reference, x)
end
function LogDensityProblems.logdensity_and_gradient(tf::PathTemperedLogDensityProblem, x)
    y, ∇y = logdensity_and_gradient(tf.logdensity, x)
    y_ref, ∇y_ref = logdensity_and_gradient(tf.reference, x)
    return tf.beta * y + (1 - tf.beta) * y_ref, tf.beta * ∇y + (1 - tf.beta) * ∇y_ref
end
