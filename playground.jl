using LabelledArrays
import LinearAlgebra
import Zygote
import FiniteDifferences

# testing different Vec3 struct definitions to figure out:
# 1. which ones are compatible with LinearAlgebra
# 2. and differentiable with Zygote or alternatives

# basic, as a struct (has no iterate)
# struct Vec3
#   x::Float64
#   y::Float64
#   z::Float64
# end

# as a NamedTuple (doesn't work with fns)
# Vec3(x, y, z) = NamedTuple{(:x, :y, :z),Tuple{Float64,Float64,Float64}}((x, y, x))

# as LabelledArrays
Vec3 = @SLVector Float64 (:x, :y, :z)
# const Vec3 = @LArray
# Zygote.@adjoint Vec3(x, y, z) = Vec3(x, y, z), p -> (p.x, p.y, p.z)

# or just use plain Vector and loose out on nice syntax

add(x::Vec3, y::Vec3) = x + y

# dot(x::Vec3, y::Vec3) = LinearAlgebra.dot((x.x, x.y, x.z), (y.x, y.y, y.z))
# dot(u::Vec3, v::Vec3) = u.x * v.x + u.y * v.y + u.z * v.z

u = Vec3(1.0, 2.0, 3.0)
v = Vec3(4.0, 5.0, 6.0)

print(LinearAlgebra.dot(u, v))
print(add(u, v))
print(Zygote.gradient(z -> add(u, z).x, v))
print(FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1; factor = 1e10), z -> add(u, z).x, v))

