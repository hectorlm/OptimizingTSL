using Base.Threads
using LinearAlgebra
using Statistics
using JLD2
using MAT

println("Job ID $(ARGS[1])")
println(nthreads())
#println(norm(randn(10,10)))
#println(mean(randn(100)))

@time @threads for i in 1:1000
	a = randn(ComplexF64, (1000,1000));
end
