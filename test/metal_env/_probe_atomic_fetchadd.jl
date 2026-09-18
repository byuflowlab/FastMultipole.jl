include(joinpath(@__DIR__, "ka_backend.jl"))
using KernelAbstractions
const KA = KernelAbstractions
dev_functional() || (println("no dev"); exit(0))

# Does `KA.@atomic c[i] += 1` return the NEW value in expression position?
# The counting-sort scatter needs slot = old+1 == new.
@kernel function claim_kernel!(perm, sorted_keys, cursor, @Const(keys))
    i = @index(Global)
    @inbounds begin
        key = keys[i]
        slot = KA.@atomic cursor[Int(key) + 1] += Int32(1)
        perm[Int(slot)] = i
        sorted_keys[Int(slot)] = key
    end
end

n, nk = 4096, 8
ks = rand(0:(nk-1), n)
counts = [count(==(k), ks) for k in 0:(nk-1)]
excl = Int32[sum(counts[1:k-1]) for k in 1:nk]
kd = KA.allocate(DEV_BACKEND, UInt64, n); copyto!(kd, UInt64.(ks))
cur = KA.allocate(DEV_BACKEND, Int32, nk); copyto!(cur, excl)
sd = KA.allocate(DEV_BACKEND, UInt64, n); copyto!(sd, zeros(UInt64, n))
pd = KA.allocate(DEV_BACKEND, Int, n); copyto!(pd, zeros(Int, n))
claim_kernel!(DEV_BACKEND, 64)(pd, sd, cur, kd; ndrange=n)
KA.synchronize(DEV_BACKEND)
p = Array(pd); s = Array(sd)
println("perm is a permutation of 1:n      : ", sort(p) == collect(1:n))
println("sorted_keys nondecreasing         : ", issorted(s))
println("sorted_keys match key multiset    : ", sort(Int.(s)) == sort(ks))
println("perm consistent with keys         : ", all(Int(s[j]) == ks[p[j]] for j in 1:n))
