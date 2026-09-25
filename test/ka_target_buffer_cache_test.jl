# The per-system device scatter buffer cache in the KernelAbstractions
# extension holds a grow-only capacity buffer and serves the live body count
# as a column-prefix view (052 long-run leak, job 13508681): a shedding run
# changes the body count every step, and an exact-size cache reallocated a
# device buffer per step that the host GC never collected. Exercised on the
# CPU backend, where the contract is the same.
using KernelAbstractions
const KAExt = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

@testset "KA target buffer cache: grow-only capacity" begin
    be = KernelAbstractions.CPU()
    cache = Dict{Int,Any}()
    b1 = KAExt._ka_cached_target_buffer(cache, be, 1, Float64, 3, 10)
    @test size(b1) == (3, 10) && !(b1 isa SubArray)          # first request: exact
    p1 = cache[1]
    b2 = KAExt._ka_cached_target_buffer(cache, be, 1, Float64, 3, 8)
    @test size(b2) == (3, 8) && parent(b2) === p1             # shrink: prefix view, no allocation
    b3 = KAExt._ka_cached_target_buffer(cache, be, 1, Float64, 3, 11)
    @test size(b3) == (3, 11) && cache[1] !== p1 && size(cache[1], 2) == 13   # grow: 10 + ceil(10/4) headroom
    p3 = cache[1]
    b4 = KAExt._ka_cached_target_buffer(cache, be, 1, Float64, 3, 13)
    @test parent(b4) === p3 || b4 === p3                        # fits the headroom: no allocation
    b5 = KAExt._ka_cached_target_buffer(cache, be, 1, Float64, 3, 12)
    @test parent(b5) === p3 && size(b5) == (3, 12)
    b6 = KAExt._ka_cached_target_buffer(cache, be, 1, Float64, 4, 5)
    @test size(cache[1]) == (4, 5)                             # a row change is a fresh exact buffer
    b7 = KAExt._ka_cached_target_buffer(cache, be, 2, Float32, 3, 7)
    @test eltype(b7) == Float32 && size(b7) == (3, 7)          # per-system, per-eltype
    @test size(KAExt._ka_cached_target_buffer(nothing, be, 1, Float64, 3, 6)) == (3, 6)   # no cache: plain allocate
end
