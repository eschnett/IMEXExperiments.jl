module IMEX



# Some explicit RK integrators

function rk2(u::S, f::Function, t::T, dt::T) where {S<:Number, T<:Real}
    t1 = t
    u1 = u
    f1 = f(t1, u1)

    t2 = t + dt
    u2 = u + dt/2 * f1
    f2 = f(t2, u2)

    t + dt, u + dt * f2
end

function rk_ssp2_222(u::S, f::Function, t::T, dt::T) where
        {S<:Number, T<:Real}
    t1 = t
    u1 = u
    f1 = f(t1, u1)

    t2 = t + dt
    u2 = u + dt * f1
    f2 = f(t2, u2)

    t + dt, u + dt/2 * f1 + dt/2 * f2
end

function rk_ssp3_332(u::S, f::Function, t::T, dt::T) where
        {S<:Number, T<:Real}
    t1 = t
    u1 = u
    f1 = f(t1, u1)

    t2 = t + dt
    u2 = u + dt * f1
    f2 = f(t2, u2)

    t3 = t + dt/2
    u3 = u + dt/4 * f1 + dt/4 * f2
    f3 = f(t3, u3)

    t + dt, u + dt/6 * f1 + dt/6 * f2 + dt*2/3 * f3
end



# Some IMEX RK integrators

# Notation: SSP_k(s, σ, p) where
#     k: order of SSP scheme
#     s: number of stages of the implicit scheme
#     σ: number of stages of the explicit scheme
#     p: order of the IMEX scheme

function imex_ssp2_222(u::S, f::Function, r::Function, t::T, dt::T) where
        {S<:Number, T<:Real}
    γ = 1 - 1/√T(2)

    t1 = t
    t1′ = t + γ * dt
    # u1 = u + γ * dt * r(t1′, u1)
    u1, r1 = r(t1′, u, γ * dt)
    f1 = f(t1, u1)

    t2 = t + dt
    t2′ = t + (1 - γ) * dt
    # u2 = u + dt * f1 + (1 - 2γ) * dt * r1 + (1 - γ) * dt * r(t + γ * dt, u2)
    u2, r2 = r(t2′, u + dt * f1 + (1 - 2γ) * dt * r1, γ * dt)
    f2 = f(t2, u2)

    t + dt, u + dt/2 * f1 + dt/2 * f2 + dt/2 * r1 + dt/2 * r2
end



# A coupled harmonic oscillator as example system

struct State{T<:Number} <: Number
    x1::T
    v1::T
    x2::T
    v2::T
end

function map1(f::Function, s::State{T})::State{T} where {T}
    State{T}(f(s.x1), f(s.v1), f(s.x2), f(s.v2))
end

function map2(f::Function, s1::State{T}, s2::State{T})::State{T} where {T}
    State{T}(f(s1.x1, s2.x1), f(s1.v1, s2.v1), f(s1.x2, s2.x2), f(s1.v2, s2.v2))
end

import Base: +, -, *, /
+(s::State{T}) where {T<:Number} = map1(+, s)
-(s::State{T}) where {T<:Number} = map1(-, s)

+(s::State{T}, x::T) where {T<:Number} = map1(y->y+x, s)
-(s::State{T}, x::T) where {T<:Number} = map1(y->y-x, s)
*(s::State{T}, x::T) where {T<:Real} = map1(y->y*x, s)
/(s::State{T}, x::T) where {T<:Real} = map1(y->y/x, s)

+(x::T, s::State{T}) where {T<:Number} = map1(y->x+y, s)
-(x::T, s::State{T}) where {T<:Number} = map1(y->x-y, s)
*(x::T, s::State{T}) where {T<:Real} = map1(y->x*y, s)
/(x::T, s::State{T}) where {T<:Real} = map1(y->x/y, s)

+(s1::State{T}, s2::State{T}) where {T<:Number} = map2(+, s1, s2)
-(s1::State{T}, s2::State{T}) where {T<:Number} = map2(-, s1, s2)
*(s1::State{T}, s2::State{T}) where {T<:Real} = map2(*, s1, s2)
/(s1::State{T}, s2::State{T}) where {T<:Real} = map2(/, s1, s2)

function output(n::Integer, t::T, s::State{T}) where {T}
    global outfile
    e = energy(s)
    println("$n $t $e")
    println(outfile, "$n $t $(s.x1) $(s.v1) $(s.x2) $(s.v2) $e")
end



const omega1 = Float64(1)
const omega2 = Float64(2)
const alpha1 = Float64(0)
const alpha2 = Float64(1000)

function initial(::Type{T}) where {T<:Real}
    t::T = 0
    x1 = -1
    v1 = 0
    x2 = -1
    v2 = 0
    t::T, State{T}(x1, v1, x2, v2)
end

function rhs(t::T, s::State{T}) where {T<:Real}
    x1 = s.v1                   # + alpha1 * (s.x2 - s.x1)
    v1 = - omega1^2 * s.x1
    x2 = s.v2                   # + alpha2 * (s.x1 - s.x2)
    v2 = - omega2^2 * s.x2
    State{T}(x1, v1, x2, v2)
end

function irhs(t::T, s0::State{T}, a::T) where {T<:Real}
    # u1 = u0 + a * r(t1, u1)
    # u1, r1 = r(t1, u0, a)

    # x1 = alpha1 * (s.x2 - s.x1)
    # v1 = 0
    # x2 = alpha2 * (s.x1 - s.x2)
    # v2 = 0
    # r = State{T}(x1, v1, x2, v2)
    # s = s0 + a * r

    offset = a * (alpha2 * s0.x1 + alpha1 * s0.x2)
    denom = 1 + a * (alpha1 + alpha2)

    x1 = (s0.x1 + offset) / denom
    v1 = s0.v1
    x2 = (s0.x2 + offset) / denom
    v2 = s0.v2
    s = State{T}(x1, v1, x2, v2)

    x1rhs = - alpha1 * (s0.x1 - s0.x2) / denom
    v1rhs = 0
    x2rhs = - alpha2 * (s0.x2 - s0.x1) / denom
    v2rhs = 0
    r = State{T}(x1rhs, v1rhs, x2rhs, v2rhs)

    s, r
end

function energy(s::State{T})::T where {T<:Real}
    e1 = T(1)/2 * omega1 * s.x1^2 + T(1)/2 / omega1 * s.v1^2
    e2 = T(1)/2 * omega2 * s.x2^2 + T(1)/2 / omega2 * s.v2^2
    e1 + e2
end



# Driver

function evolve()
    global outfile = open("harmonic.asc", "w")
    println(outfile, "# Iteration time x1 v1 x2 v2 e")

    odeint = rk2
    # odeint = rk_ssp2_222
    # odeint = rk_ssp3_332
    odeimex = imex_ssp2_222

    t::Float64, s::State{Float64} = initial(Float64)
    output(0, t, s)
    niters = 1000
    dt::Float64 = 0.05
    for n in 1:niters
        # t, s = odeint(s, rhs, t, dt)
        t, s = odeimex(s, rhs, irhs, t, dt)
        output(n, t, s)
    end

    close(outfile)
end

end
