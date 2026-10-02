# rigid-motion helper for the transform_tree!/transform_plan!/transform_solver! tests
# Rodrigues rotation about unit axis n by angle theta
function _rodrigues(n::SVector{3,Float64}, theta::Float64)
    n = n / norm(n)
    K = SMatrix{3,3,Float64,9}(0, n[3], -n[2], -n[3], 0, n[1], n[2], -n[1], 0)
    return SMatrix{3,3,Float64,9}(I) + sin(theta) * K + (1 - cos(theta)) * K * K
end
