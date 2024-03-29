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
LogDensityProblems.logdensity(tf::TemperedLogDensityProblem, x) = tf.beta * LogDensityProblems.logdensity(tf.logdensity, x)
function LogDensityProblems.logdensity_and_gradient(tf::TemperedLogDensityProblem, x)
    y, ∇y = LogDensityProblems.logdensity_and_gradient(tf.logdensity, x)
    return tf.beta .* y, tf.beta .* ∇y
end

