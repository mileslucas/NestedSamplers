module NestedSamplers

using Random
using LinearAlgebra
using Distributions: Distribution,
                     quantile,
                     cdf
using MCMCChains: Chains
import AbstractMCMC: AbstractSampler,
                     AbstractModel,
                     step!,
                     sample_init!,
                     sample_end!,
                     bundle_samples
using StatsFuns: logaddexp,
                 log1mexp

export NestedModel,
       Nested,
       dlogz_convergence,
       decline_covergence


include("ellipsoids.jl")

###############################################################################
# Interface Implementations

mutable struct Nested{E <: AbstractEllipsoid,T} <: AbstractSampler
    nactive::Integer
    enlarge::Float64
    update_interval::Integer

    # behind the scenes things
    active_points::Matrix{T}
    active_logl::Vector{T}
    active_ell::E
    logz::Float64
    h::Float64
    ndecl::Integer
end

"""
    Nested(nactive, enlarge=1.2; update_interval=round(Int, 0.6nactive), method=:single)

Nested Sampler

The two `NestedAlgorithm`s are `:single`, which uses a single bounding ellipsoid, and `:multi`, which finds an optimal clustering of ellipsoids.
"""
function Nested(nactive, enlarge = 1.2; update_interval = round(Int, 0.6nactive), method = :single)
    if method === :single
        ell = Ellipsoid(1)
    elseif method === :multi
        ell = MultiEllipsoid([Ellipsoid(1)])
    else
        error("Invalid method $method")
    end
    #= Note: initializing logz as -Inf causes ugly failures in the h calculations
    by setting to a very small value (even smaller than log(eps(Float64))) we avoid this issue =#
    return Nested(nactive, enlarge, update_interval, zeros(0, nactive), zeros(nactive), ell, -1e300, 0.0, 0)
end

function Base.show(io::IO, n::Nested)
    println(io, "Nested{$(typeof(n.active_ell))}(nactive=$(n.nactive), enlarge=$(n.enlarge), update_interval=$(n.update_interval))")
    println(io, "  logz=$(n.logz) +- $(sqrt(n.h / n.nactive))")
    print(io,   "  h=$(n.h)")
end

struct NestedModel{F <: Function,D <: Distribution} <: AbstractModel
    loglike::F
    priors::Vector{D}
end

struct NestedTransition{T}
    draw::Vector{T}  # the sample
    logL::Float64    # log likelihood
    log_vol::Float64 # log mass of objects in hyperspace
    log_wt::Float64  # log weight of this draw
end

function sample_init!(rng::AbstractRNG,
    model::NestedModel,
    s::Nested{E},
    ::Integer;
    debug::Bool = false,
    kwargs...) where {E <: AbstractEllipsoid}

    debug && @info "Initializing sampler"
    s.nactive < 2length(model.priors) && @warn "Using fewer than 2ndim active points is discouraged"

    # samples in unit space
    us = rand(rng, length(model.priors), s.nactive)

    # samples and loglikes in prior space
    s.active_points = quantile.(hcat(model.priors), us)
    s.active_logl = [model.loglike(s.active_points[:, i]) for i in 1:s.nactive]

    # get bounding ellipsoid
    s.active_ell = scale!(fit(E, us, pointvol = 1 / s.nactive), s.enlarge)
end

function step!(rng::AbstractRNG,
    model::NestedModel,
    s::Nested,
    ::Integer;
    kwargs...)
    # Find least likely point
    logL, idx = findmin(s.active_logl)
    draw = s.active_points[:, idx]

    # Initial point will have volume 1 - exp(-1/npoints)
    log_vol = log1mexp(-1 / s.nactive)
    log_wt = log_vol + logL

    return NestedTransition(draw, logL, log_vol, log_wt)
end

function step!(rng::AbstractRNG,
    model::NestedModel,
    s::Nested{E},
    ::Integer,
    prev::NestedTransition;
    iteration,
    debug::Bool = false,
    kwargs...) where {E <: AbstractEllipsoid}

    # update sampler
    logz = logaddexp(s.logz, prev.log_wt)
    s.h = (exp(prev.log_wt - logz) * prev.logL +
        exp(s.logz - logz) * (s.h + s.logz) - logz)

    s.logz = logz

    # Get bounding ellipsoid (only every update_interval)
    if iteration % s.update_interval == 0
        # Get points in unit space
        u = cdf.(hcat(model.priors), s.active_points)

        # fit ellipsoid
        pointvol = exp(-(iteration - 1) / s.nactive) / s.nactive
        s.active_ell = scale!(fit(E, u, pointvol = pointvol), s.enlarge)
    end

    # Find least likely point
    logL, idx = findmin(s.active_logl)

    draw = s.active_points[:, idx]

    log_vol = prev.log_vol - 1 / s.nactive
    log_wt = log_vol + logL

    # Get new point and log like
    p, logl = propose(rng, s.active_ell, model, logL)
    s.active_points[:, idx] = p
    s.active_logl[idx] = logl
    s.ndecl = log_wt < prev.log_wt ? s.ndecl + 1 : 0

    return NestedTransition(draw, logL, log_vol, log_wt)
end

function sample_end!(rng::AbstractRNG,
    ℓ::AbstractModel,
    s::Nested,
    ::Integer,
    transitions;
    debug::Bool = false,
    kwargs...)
    # Pop remaining points in ellipsoid
    N = length(transitions)
    log_vol = -N / s.nactive - log(s.nactive)
    @inbounds for i in eachindex(s.active_logl)
        # get new point
        draw = s.active_points[:, i]
        logL = s.active_logl[i]
        log_wt = log_vol + logL

        # update sampler
        logz = logaddexp(s.logz, log_wt)
        s.h = (exp(log_wt - logz) * logL +
               exp(s.logz - logz) * (s.h + s.logz) - logz)
        s.logz = logz

        prev = NestedTransition(draw, logL, log_vol, log_wt)
        push!(transitions, prev)
    end

    # h should always be non-negative. Numerical error can arise from pathological corner cases
    if s.h < 0
        s.h < -√eps(s.h) && @warn "Negative h encountered h=$(s.h). This is likely a bug"
        s.h = zero(s.h)
    end
end

function bundle_samples(rng::AbstractRNG,
    ℓ::AbstractModel,
    s::Nested,
    N::Integer,
    transitions,
    Chains;
    param_names = missing,
    check_wsum = true,
    kwargs...)

    vals = copy(mapreduce(t->vcat(t.draw, t.log_wt), hcat, transitions)')
    # update weights based on evidence
    @. vals[:, end, 1] = exp(vals[:, end, 1] - s.logz)
    wsum = sum(vals[:, end, 1])
    @. vals[:, end, 1] /= wsum

    if check_wsum
        err = !iszero(s.h) ? 3sqrt(s.h / s.nactive) : 1e-3
        isapprox(wsum, 1, atol = err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    # Parameter names
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(vals[1, :]) - 1]
    end
    push!(param_names, "weights")

    return Chains(vals, param_names, Dict(:internals => ["weights"]), evidence = exp(s.logz))
end

function bundle_samples(rng::AbstractRNG,
    ℓ::AbstractModel,
    s::Nested,
    N::Integer,
    transitions,
    ::Union{AbstractArray,Any};
    check_wsum = true,
    kwargs...)

    vals = copy(mapreduce(t->t.draw, hcat, transitions)')
    
    if check_wsum
        # get weights
        wsum = mapreduce(t->exp(t.log_wt - s.logz), +, transitions)

        # check with h
        err = s.h ≠ 0 ? 3sqrt(s.h / s.nactive) : 1e-3
        isapprox(wsum, 1, atol = err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    return vals
end

"""
Propose a new point in the given `AbstractEllipsoid` that is guaranteed to have log-likelihood greater than or equal to `logl_star`
"""
function propose(rng::AbstractRNG, ell::AbstractEllipsoid, model::NestedModel, logl_star)
    while true
        u = rand(rng, ell)
        all(0 .< u .< 1) || continue
        v = quantile.(model.priors, u)
        logl = model.loglike(v)
        logl ≥ logl_star && return v, logl
    end
end

###############################################################################
# Convergence methods

"""
Stopping criterion: Number of consecutive declining log-evidence is greater than `decline_factor` * `iteration`
"""
function decline_covergence(rng::AbstractRNG,
    model::AbstractModel,
    s::Nested,
    transitions,
    iteration::Integer;
    decline_factor = 1,
    kwargs...)
    return s.ndecl > decline_factor * iteration
end

"""
Stopping criterion: estimated fraction evidence remaining below threshold
"""
function dlogz_convergence(rng::AbstractRNG,
    model::AbstractModel,
    s::Nested,
    transitions,
    iteration::Integer;
    dlogz = 0.5,
    kwargs...)
    logz_remain = maximum(s.active_logl) - (iteration - 1) / s.nactive
    dlogz_current = logaddexp(s.logz, logz_remain) - s.logz
    return dlogz_current < dlogz
end

end
