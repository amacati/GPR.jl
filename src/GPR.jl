module GPR

using StaticArrays
using LinearAlgebra
using Statistics
using Plots
using Optim
using ConstrainedDynamicsVis
using Rotations

export GaussianProcessRegressor
export GaussianProcessQuaternionRegressor
export MOGaussianProcessRegressor
export AbstractKernel
export GaussianKernel
export GeneralGaussianKernel
export MaternKernel
export QuaternionKernel
export CompositeKernel
export predict
export predict_full
export plot_gp
export optimize!
export visualize_prediction
export compute_kernelmatrix
export quaternion_average
export quaternion_to_array
export quaternion_projection
export updatestate

include(joinpath("kernels", "AbstractKernel.jl"))
include(joinpath("utils", "functions.jl"))
include(joinpath("utils", "buffer.jl"))
include(joinpath("visualization", "visualization.jl"))
include(joinpath("kernels", "GaussianKernel.jl"))
include(joinpath("kernels", "GeneralGaussianKernel.jl"))
include(joinpath("kernels", "MaternKernel.jl"))
include(joinpath("kernels", "QuaternionKernel.jl"))
include(joinpath("kernels", "CompositeKernel.jl"))
include(joinpath("utils", "kernelmatrix.jl"))
include(joinpath("regressors", "gp_regressor.jl"))
include(joinpath("regressors", "gp_quat_regressor.jl"))
include(joinpath("regressors", "mogp_regressor.jl"))
include(joinpath("utils", "optimization.jl"))
end