"""
    unzip

Standard solution to inverting the zip operation, turn a collection of lists of items of varying type into a collection of lists of each type of item
i.e. [[1, a, A], [2, b, B]] -> [[1, 2], [a, b], [A, B]]
- `collection` of items to unzip and return
"""
function unzip(collection)
    # collect(zip(collection...))
    [[x[i] for x in collection] for i in 1:length(collection[1])]
end


"""
    logdensity

Calls appropriate log-density function for a given sampler
"""
function logdensity(
    sampler,
    model,
    temperature
)

end