using Images
using LabelledArrays
using LinearAlgebra
using ColorVectorSpace
using CMAEvolutionStrategy
import Zygote
import ChainRulesCore
import FiniteDifferences
import Random
import ImageDistances
import GalacticOptim
import Optim
import NLopt
import Evolutionary

Threads.nthreads()

Vec3 = @SLVector Float64 (:x, :y, :z)

# struct Vec3
#   x::Float64
#   y::Float64
#   z::Float64
# end

# Base.:+(u::Vec3, v::Vec3) = Vec3(u.x + v.x, u.y + v.y, u.z + v.z)
# Base.:-(v::Vec3) = Vec3(-v.x, -v.y, -v.z)
# Base.:-(u::Vec3, v::Vec3) = Vec3(u.x - v.x, u.y - v.y, u.z - v.z)
Base.:*(u::Vec3, v::Vec3) = u .* v
# Base.:*(u::Vec3, v::Vec3) = Vec3(u.x * v.x, u.y * v.y, u.z * v.z)
# Base.:*(v::Vec3, t::Number) = Vec3(v.x * t, v.y * t, v.z * t)
# Base.:*(t::Number, v::Vec3) = v * t
# Base.:/(v::Vec3, t::Number) = Vec3(v.x / t, v.y / t, v.z / t)

length_squared(v::Vec3) = v.x * v.x + v.y * v.y + v.z * v.z

# length(collection) is for number of elements
length_vector(v::Vec3) = sqrt(length_squared(v))

# dot(u::Vec3, v::Vec3) = u.x * v.x + u.y * v.y + u.z * v.z

# cross(u::Vec3, v::Vec3) = Vec3(
#   u.y * v.z - u.z * v.y,
#   u.z * v.x - u.x * v.z,
#   u.x * v.y - u.y * v.x
# )

unit_vector(v::Vec3) = v / length_vector(v)

reflect(v::Vec3, n::Vec3) = v - 2 * dot(v, n) * n

function refract(uv::Vec3, n::Vec3, etai_over_etat::Float64)::Vec3
  cos_theta = dot(-uv, n)
  r_out_parallel = etai_over_etat * (uv + cos_theta * n)
  r_out_perp = -sqrt(1 - length_squared(r_out_parallel)) * n
  r_out_parallel + r_out_perp
end

function random_in_unit_sphere()::Vec3
  while true
    p = Vec3(rand(-1:eps():1), rand(-1:eps():1), rand(-1:eps():1))
    if length_squared(p) < 1
      return p
    end
  end
end

function random_unit_vector()::Vec3
  a = rand(0:eps():2pi)
  z = rand(-1:eps():1)
  r = sqrt(1 - z * z)
  Vec3(r * cos(a), r * sin(a), z)
end

function random_in_hemisphere(normal::Vec3)::Vec3
  in_unit_sphere = random_in_unit_sphere()
  if dot(in_unit_sphere, normal) > 0
    in_unit_sphere
  else
    -in_unit_sphere
  end
end

function random_in_unit_disk()::Vec3
  while true
    p = Vec3(rand(-1:eps():1), rand(-1:eps():1), 0)
    if length_squared(p) < 1
      return p
    end
  end
end

struct Ray
  origin::Vec3
  direction::Vec3
end

at(ray::Ray, t::Number)::Vec3 = ray.origin + t * ray.direction

abstract type Hittable end

abstract type Material end

struct HitRecord
  p::Vec3
  normal::Vec3
  mat::Material
  t::Float64
  front_face::Bool

  function HitRecord(
    p::Vec3,
    t::Number,
    mat::Material,
    r::Ray,
    outward_normal::Vec3)
    front_face = dot(r.direction, outward_normal) < 0
    normal = front_face ? outward_normal : -outward_normal
    new(p, normal, mat, t, front_face)
  end
end

struct Sphere <: Hittable
  center::Vec3
  radius::Float64
  mat::Material
end

function hit(
  sphere::Sphere,
  r::Ray,
  t_min::Number,
  t_max::Number,
)::Union{HitRecord,Nothing}

  oc = r.origin - sphere.center
  a = length_squared(r.direction)
  half_b = dot(oc, r.direction)
  c = length_squared(oc) - sphere.radius * sphere.radius
  discriminant = half_b * half_b - a * c

  if discriminant > 0
    root = sqrt(discriminant)
    temp = (-half_b - root) / a
    if t_min < temp < t_max
      p = at(r, temp)
      outward_normal = (p - sphere.center) / sphere.radius
      HitRecord(p, temp, sphere.mat, r, outward_normal)
    else
      temp = (-half_b + root) / a
      if t_min < temp < t_max
        p = at(r, temp)
        outward_normal = (p - sphere.center) / sphere.radius
        HitRecord(p, temp, sphere.mat, r, outward_normal)
      end
    end
  end
end

struct HittableList <: Hittable
  objects::Vector{Sphere}
end

function hit(
  hl::HittableList,
  r::Ray,
  t_min::Number,
  t_max::Number,
)::Union{HitRecord,Nothing}
  closest_so_far = t_max

  hr::Union{HitRecord,Nothing} = nothing

  for obj in hl.objects
    temp_hr = hit(obj, r, t_min, closest_so_far)
    if temp_hr != nothing
      closest_so_far = temp_hr.t
      hr = temp_hr
    end
  end

  hr
end

struct ScatterRecord
  attenuation::Vec3
  scattered::Ray
end

struct Lambertian <: Material
  albedo::Vec3
end

function scatter(
  lambertian::Lambertian,
  r_in::Ray,
  rec::HitRecord,
)::Union{ScatterRecord,Nothing}
  scatter_direction = rec.normal + random_unit_vector()
  scattered = Ray(rec.p, scatter_direction)
  attenuation = lambertian.albedo
  ScatterRecord(attenuation, scattered)
end

struct Metal <: Material
  albedo::Vec3
  fuzz::Float64

  Metal(albedo::Vec3, fuzz::Number) = new(albedo, fuzz < 1 ? fuzz : 1)
end

function scatter(
  metal::Metal,
  r_in::Ray,
  rec::HitRecord)::Union{ScatterRecord,Nothing}
  reflected = reflect(unit_vector(r_in.direction), rec.normal)
  scattered = Ray(rec.p, reflected)
  attenuation = metal.albedo
  if dot(scattered.direction, rec.normal) > 0
    ScatterRecord(attenuation, scattered)
  end
end

function schlick(cosine::Float64, ref_idx::Float64)::Float64
  r0 = (1 - ref_idx) / (1 + ref_idx)
  r0 *= r0
  r0 + (1 - r0) * (1 - cosine)^5
end

struct Dielectric <: Material
  ref_idx::Float64
end

function scatter(
  dielectric::Dielectric,
  r_in::Ray,
  rec::HitRecord
)::Union{ScatterRecord,Nothing}
  attenuation = Vec3(1, 1, 1)
  etai_over_etat =
    rec.front_face ? 1 / dielectric.ref_idx : dielectric.ref_idx

  unit_direction = unit_vector(r_in.direction)
  cos_theta = min(dot(-unit_direction, rec.normal), 1)
  sin_theta = sqrt(1 - cos_theta * cos_theta)

  if etai_over_etat * sin_theta > 1
    reflected = reflect(unit_direction, rec.normal)
    scattered = Ray(rec.p, reflected)
    ScatterRecord(attenuation, scattered)
  else
    reflect_prob = schlick(cos_theta, etai_over_etat)
    if rand() < reflect_prob
      reflected = reflect(unit_direction, rec.normal)
      scattered = Ray(rec.p, reflected)
      ScatterRecord(attenuation, scattered)
    else
      refracted = refract(unit_direction, rec.normal, etai_over_etat)
      scattered = Ray(rec.p, refracted)
      ScatterRecord(attenuation, scattered)
    end
  end
end

struct Camera
  origin::Vec3
  lower_left_corner::Vec3
  horizontal::Vec3
  vertical::Vec3
  u::Vec3
  v::Vec3
  w::Vec3
  lens_radius::Float64

  function Camera(
    lookfrom::Vec3,
    lookat::Vec3,
    vup::Vec3,
    vfov::Number,
    aspect_ratio::Number,
    aperture::Number,
    focus_dist::Number)
    origin = lookfrom
    lens_radius = aperture / 2

    theta = deg2rad(vfov)
    half_height = tan(theta / 2)
    half_width = aspect_ratio * half_height

    w = unit_vector(lookfrom - lookat)
    u = unit_vector(cross(vup, w))
    v = cross(w, u)

    lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w

    horizontal = 2 * half_width * focus_dist * u
    vertical = 2 * half_height * focus_dist * v

    new(origin, lower_left_corner, horizontal, vertical, u, v, w, lens_radius)
  end
end

function get_ray(camera::Camera, s::Number, t::Number)::Ray
  rd = camera.lens_radius * random_in_unit_disk()
  offset = camera.u * rd.x + camera.v * rd.y

  Ray(
    camera.origin + offset,
    camera.lower_left_corner + s * camera.horizontal + t * camera.vertical - camera.origin - offset)
end

function random_scene(seed)
  Random.seed!(seed)
  world = HittableList([])
  push!(world.objects, Sphere(Vec3(0, -1000, 0), 1000, Lambertian(Vec3(0.5, 0.5, 0.5))))

  for a in -11:10, b in -11:10
    choose_mat = rand()
    center = Vec3(a + 0.9 * rand(), 0.2, b + 0.9 * rand())

    if length_vector(center - Vec3(4, 0.2, 0)) > 0.9
      if choose_mat < 0.8
        # diffuse
        albedo = Vec3(rand(), rand(), rand()) * Vec3(rand(), rand(), rand())
        push!(world.objects, Sphere(center, 0.2, Lambertian(albedo)))
      elseif choose_mat < 0.95
        # metal
        albedo = Vec3(rand(0.5:eps():1), rand(0.5:eps():1), rand(0.5:eps():1))
        fuzz = rand(0:eps():0.5)
        push!(world.objects, Sphere(center, 0.2, Metal(albedo, fuzz)))
      else
        # glass
        push!(world.objects, Sphere(center, 0.2, Dielectric(1.5)))
      end
    end
  end

  push!(world.objects, Sphere(Vec3(0, 1, 0), 1, Dielectric(1.5)))
  push!(world.objects, Sphere(Vec3(-4, 1, 0), 1, Lambertian(Vec3(0.4, 0.2, 0.1))))
  push!(world.objects, Sphere(Vec3(4, 1, 0), 1, Metal(Vec3(0.7, 0.6, 0.5), 0.0)))

  world
end

function ray_color(world::HittableList, r::Ray, depth::Int)::Vec3
  if depth <= 0
    return Vec3(0, 0, 0)
  end

  hr = hit(world, r, 0.001, Inf)

  if hr != nothing
    sr = scatter(hr.mat, r, hr)
    if sr != nothing
      sr.attenuation * ray_color(world, sr.scattered, depth - 1)
    else
      Vec3(0, 0, 0)
    end
  else
    unit_direction = unit_vector(r.direction)
    t = 0.5 * (unit_direction.y + 1)
    (1 - t) * Vec3(1, 1, 1) + t * Vec3(0.5, 0.7, 1)
  end
end

function color(world::HittableList, cam, i::Int, j::Int)
  c = Vec3(0, 0, 0)
  for _ = 1:samples_per_pixel
    u = (i + rand()) / image_width
    v = (j + rand()) / image_height
    r = get_ray(cam, u, v)
    c += ray_color(world, r, max_depth)
  end
  c /= samples_per_pixel
  RGB(sqrt(c.x), sqrt(c.y), sqrt(c.z))
  # c
end

function render(cam, seed)
  # ensure global random seed for the whole render
  Random.seed!(seed)
  world = random_scene(seed)
  #     @threaded 
  map(coords -> color(world, cam, coords...), [(i, j) for j in image_height:-1:1, i in 1:image_width])
end

const aspect_ratio = 16 / 9
const image_width = 10
const image_height = Int(image_width ÷ aspect_ratio)
const samples_per_pixel = 1
const max_depth = 2;
const lookfrom = Vec3(13, 2, 3)
const lookat = Vec3(0, 0, 0)
const vup = Vec3(0, 1, 0)
const dist_to_focus = 10
const aperture = 0.1

const cam1 = Camera(lookfrom, lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus);
a = render(cam1, 1234)

# get gradients, camera lookfrom is different between the 2 renders, loss is mse
# note random seed is the same, in reality that would provide additonal noise
# slightly incorrect:
FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), lookfromB -> ImageDistances.mse(a, render(Camera(lookfromB, lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus), 1234)), Vec3(13, 2, 3))

# never completes:
# Zygote.gradient(lookfromB -> ImageDistances.mse(a, render(Camera(lookfromB, lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus), 1234)), Vec3(13, 0, 3))

ImageDistances.mse(a, render(Camera(Vec3(13, 0, 3), lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus), 1234))
# @time c = render(Camera(Vec3(13, 2, 3), lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus), 1234)

lookfromB = [13.0, 0.0, 3.0]
renderB(lookfromB, p) = ImageDistances.mse(a, render(Camera(Vec3(lookfromB[1], lookfromB[2], lookfromB[3]), lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus), 1234))

prob = GalacticOptim.OptimizationProblem(renderB, lookfromB)
sol = GalacticOptim.solve(prob, Optim.NelderMead())

# todo figure out AutoForwardDiff
# renderBF = GalacticOptim.OptimizationFunction(renderB, GalacticOptim.AutoForwardDiff())
# prob = GalacticOptim.OptimizationProblem(renderBF, lookfromB)
# sol = GalacticOptim.solve(prob, Optim.IPNewton())

prob = GalacticOptim.OptimizationProblem(renderB, lookfromB)
sol = GalacticOptim.solve(prob, GalacticOptim.CMAEvolutionStrategyOpt())

# this one finds the answer if given lower and upper bounds!
prob = GalacticOptim.OptimizationProblem(renderB, lookfromB, lb = [0, 0, 0], ub = [14, 14, 14])
sol = GalacticOptim.solve(prob, NLopt.LN_NELDERMEAD())

prob = GalacticOptim.OptimizationProblem(renderB, lookfromB)
sol = GalacticOptim.solve(prob, Evolutionary.CMAES(metrics = [Evolutionary.AbsDiff(1e-5)]), iterations = 10000000000)

prob = GalacticOptim.OptimizationProblem(renderB, lookfromB, lb = [0, 0, 0], ub = [14, 14, 14])
sol = GalacticOptim.solve(prob, Evolutionary.CMAES(metrics = [Evolutionary.AbsDiff(1e-5)]))
