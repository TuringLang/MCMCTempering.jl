"""
    unzip

Standard solution to inverting the zip operation, turn a collection of lists of items of varying type into a collection of lists of each type of item
i.e. [[1, a, A], [2, b, B]] -> [[1, 2], [a, b], [A, B]]
- `collection` of items to unzip and return
"""
function unzip(collection)
    collect(zip(collection...))
end