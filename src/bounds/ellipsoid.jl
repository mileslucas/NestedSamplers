"""
    Bounds.Ellipsoid([T=Float64], N)
    Bounds.Ellipsoid(center::AbstractVector, A::AbstractMatrix)

An `N`-dimensional ellipsoid defined by

```math
(x - center)^T A (x - center) = 1
```

where `size(center) == (N,)` and `size(A) == (N,N)`.

This implementation follows the algorithm presented in Mukherjee et al. (2006).[^2]

[^2]: Pia Mukherjee, et al., 2006, ApJ 638 L51 ["A Nested Sampling Algorithm for Cosmological Model Selection"](https://iopscience.iop.org/article/10.1086/501068)
"""
mutable struct Ellipsoid{T} <: AbstractBoundingSpace{T}
    center::Vector{T}
    A::Symmetric{T, Matrix{T}}
    axes::Matrix{T}
    axlens::Vector{T}
    volume::T
end


Ellipsoid(center::AbstractVector, A::AbstractMatrix) = Ellipsoid(center, Symmetric(A))
function Ellipsoid(center::AbstractVector, A::Symmetric)
    axes, axlens = decompose(A)
    Ellipsoid(center, A, axes, axlens, _volume(A))
end
Ellipsoid(ndim::Integer) = Ellipsoid(Float64, ndim)
Ellipsoid(T::Type, ndim::Integer) = Ellipsoid(zeros(T, ndim), diagm(0 => ones(T, ndim)))
Ellipsoid{T}(center::AbstractVector, A::AbstractMatrix) where {T} = Ellipsoid(convert(AbstractVector{T}, center), convert(AbstractMatrix{T}, A))

Base.broadcastable(e::Ellipsoid) = (e,)

Base.ndims(ell::Ellipsoid) = length(ell.center)

# Returns the volume of an ellipsoid given its axes matrix
_volume(A::AbstractMatrix{T}) where {T} = T(volume_prefactor(size(A, 1))) / sqrt(det(A))
volume(ell::Ellipsoid) = ell.volume

# Returns the principal axes
axes(ell::Ellipsoid) = ell.axes

function decompose(A::Symmetric)
    E = eigen(A)
    axlens = @. 1 / sqrt(E.values)
    axes = E.vectors * Diagonal(axlens)
    return axes, axlens
end

# axes and axlens
decompose(ell::Ellipsoid) = ell.axes, ell.axlens

# Scale to new volume
function scale!(ell::Ellipsoid, factor)
    # linear factor
    f = factor^(1 / ndims(ell))
    ell.A /= f^2
    ell.axes .*= f
    ell.axlens .*= f
    ell.volume *= factor
    return ell
end

function endpoints(ell::Ellipsoid)
    axes, axlens = decompose(ell)
        # find major axis
    major_axis = axes[:, argmax(axlens)]
    return ell.center .- major_axis, ell.center .+ major_axis
end

function Base.in(x::AbstractVector, ell::Ellipsoid)
    d = x .- ell.center
    return dot(d, ell.A * d) ≤ 1.0
end

randoffset(rng::AbstractRNG, ell::Ellipsoid{T}) where {T} = axes(ell) * randball(rng, T, ndims(ell))
Base.rand(rng::AbstractRNG, ell::Ellipsoid) = ell.center .+ randoffset(rng, ell)

fit(E::Type{<:Ellipsoid}, x::AbstractMatrix{S}; pointvol = 0) where {S} = fit(E{float(S)}, x; pointvol = pointvol)

function fit(E::Type{<:Ellipsoid{R}}, x::AbstractMatrix{S}; pointvol = 0) where {R,S}
    T = float(promote_type(R, S))
    x = T.(x)
    ndim, npoints = size(x)

    # single element is an n-sphere with pointvol volume
    if npoints == 1
        pointvol > 0 || error("Cannot compute bounding ellipsoid with one point without a valid pointvol (got $pointvol)")
        d = log(pointvol) - log(volume_prefactor(ndim))
        r = exp(d / ndim)
        A = diagm(0 => fill(1 / r^2, ndim))
        return Ellipsoid(vec(x), A)
    end
    # get estimators
    center, cov_ = mean_and_cov(x, 2)
    delta = x .- center
    # Covariance is smaller than r^2 by a factor of 1/(n+2)
    cov = Symmetric(cov_) * (ndim + 2)
    # ensure cov is posdef and get transformation matrix
    cov, A = make_eigvals_positive(cov)

    # calculate expansion factor necessary to bound each points
    f = diag(delta' * (A * delta))
    fmax = maximum(f)

    # try to avoid round-off errors s.t. furthest point obeys
    # x^T A x < 1 - √eps
    flex = 1 - sqrt(eps(T))
    if fmax > flex
        A *= flex / fmax
    end

    ell = E(vec(center), A)

    if pointvol > 0
        minvol = npoints * pointvol
        vol = volume(ell)
        vol < minvol && scale!(ell, minvol / vol)
    end

    return ell
end


# ---------------------------------------------
# Helper functions

"""
    volume_prefactor(::Integer)

Volume constant for an n-dimensional sphere:

for n even:      (2pi)^(n    /2) / (2 * 4 * ... * n)
for n odd :  2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)
"""
function volume_prefactor(n::Integer)
    f, range = iseven(n) ? (1.0, 2:2:n) : (2.0, 3:2:n)
    for i in range
        f *= 2π / i
    end
    return f
end

# sample N samples from unit D-dimensional ball
randball(T::Type, D::Integer, N::Integer) = randball(GLOBAL_RNG, T, D, N)
function randball(rng::AbstractRNG, T::Type, D::Integer, N::Integer)
    z = randn(rng, T, D, N)
    z .*= rand(rng, T, 1, N).^(1 ./ D) ./ sqrt.(sum(p->p^2, z, dims = 1))
    return z
end

# sample from unit D-dimensional ball
randball(T::Type, D::Integer) = randball(GLOBAL_RNG, T, D)
function randball(rng::AbstractRNG, T::Type, D::Integer)
    z = randn(rng, T, D)
    z .*= rand(rng)^(1 / D) / sqrt(sum(p->p^2, z))
    return z
end

function make_eigvals_positive(cov::Symmetric)
    E = eigen(cov)
    maxcond = 1e10
    maxval = max(maximum(E.values), 1/1e10)
    if minimum(E.values) / maxval < 1/maxcond
        E.values .= max.(E.values, maxval / maxcond)
        return Symmetric(Matrix(E)), Symmetric(inv(E))
    end
    return cov, Symmetric(inv(E))
end
