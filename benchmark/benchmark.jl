benchmarks = ["optimization"
              "inference"
              "predict"]

for benchmark in benchmarks
    include("b_" * benchmark * ".jl")
end
