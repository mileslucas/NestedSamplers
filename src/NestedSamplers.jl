module NestedSamplers

using LinearAlgebra
using Random
using Random: AbstractRNG, GLOBAL_RNG

using AbstractMCMC
using AbstractMCMC: AbstractSampler,
                    AbstractModel,
                    samples,
                    save!!
import AbstractMCMC: step,
                     bundle_samples,
                     mcmcsample
using Distributions: quantile, UnivariateDistribution
using LogExpFunctions: logaddexp, log1mexp
using MCMCChains: Chains
using ProgressLogging
import StatsBase


export Bounds,
       Proposals,
       Models,
       NestedModel,
       Nested

include("model.jl")         # The default model for nested sampling

# load submodules
include("bounds/Bounds.jl")
using .Bounds
include("proposals/Proposals.jl")
using .Proposals

include("staticsampler.jl") # The static nested sampler
include("step.jl")          # The stepping mechanics (extends AbstractMCMC)
include("sample.jl")        # Custom sampling (extends AbstractMCMC)

include("models/Models.jl")
using .Models

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require DynamicPPL = "366bfd00-2699-11ea-058f-f148b4cae6d8" include(
            "../ext/NestedSamplersDynamicPPLExt.jl"
        )
    end
end

end
