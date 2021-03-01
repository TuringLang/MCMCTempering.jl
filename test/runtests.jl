using Tempering
using Test

@testset "Tempering.jl" begin
    x, y = 5, 7
    @test foo(x, y) == 7
    x = "blah"
    @test_throws MethodError foo(x, y)
    z = 4.
    @test bar(z) == 1.
end
