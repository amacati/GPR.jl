benchmarks = ["inversion"
              "inference"]

for benchmark in benchmarks
    include("b_" * benchmark * ".jl")
end
