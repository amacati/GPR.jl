module GPR

using StaticArrays
using LinearAlgebra
using Statistics
using Plots
using Optim
using ConstrainedDynamicsVis
using Rotations

export GaussianProcessRegressor
export MOGaussianProcessRegressor
export AbstractKernel
export GaussianKernel
export GeneralGaussianKernel
export MaternKernel
export predict
export predict_full
export plot_gp
export optimize!
export visualize_prediction

include(joinpath("kernels", "AbstractKernel.jl"))
include(joinpath("utils", "decompositions.jl"))
include(joinpath("visualization", "visualization.jl"))
include(joinpath("kernels", "GaussianKernel.jl"))
include(joinpath("kernels", "GeneralGaussianKernel.jl"))
include(joinpath("kernels", "MaternKernel.jl"))
include(joinpath("utils", "kernelmatrix.jl"))
include(joinpath("regressors", "gp_regressor.jl"))
include(joinpath("regressors", "mogp_regressor.jl"))
include(joinpath("utils", "optimization.jl"))
end