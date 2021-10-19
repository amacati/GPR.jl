module GPR

using StaticArrays
using LinearAlgebra
using Statistics
using Plots
using Optim
using ConstrainedDynamicsVis
using Rotations

export GaussianProcessRegressor
export AbstractKernel
export GaussianKernel
export GeneralGaussianKernel
export CompositeKernel
export predict
export predict_full
export optimize!
export updatestate
export resetMechanism!
export updateMechanism!
export projectv!

export modifykernel!

include(joinpath("kernels", "AbstractKernel.jl"))
include(joinpath("projections", "implicitProjection.jl"))
include(joinpath("utils", "functions.jl"))
include(joinpath("utils", "buffer.jl"))
include(joinpath("kernels", "GaussianKernel.jl"))
include(joinpath("kernels", "GeneralGaussianKernel.jl"))
include(joinpath("kernels", "QuaternionKernel.jl"))
include(joinpath("kernels", "CompositeKernel.jl"))
include(joinpath("utils", "kernelmatrix.jl"))
include(joinpath("regressors", "gp_regressor.jl"))
include(joinpath("regressors", "gp_quat_regressor.jl"))
include(joinpath("regressors", "mogp_regressor.jl"))
include(joinpath("utils", "optimization.jl"))
end