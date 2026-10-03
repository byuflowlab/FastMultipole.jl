# Gate for the automatic concat-M2L chunk (`ConcatenatedFixedZM2L(0)`,
# ext/ka/ka_workspace_routes.jl): the free-memory rule `_ka_auto_m2l_chunk` and
# the out-of-memory retry `_ka_with_chunk_halving`.
#
# Every real GPU tried so far has had enough free memory to sit at the cap, so
# the shrinking branches never ran there; here a stand-in backend reports chosen
# free-memory sizes, and stand-in builds throw the errors the backends throw
# (Metal: Base.OutOfMemoryError; CUDA: OutOfGPUMemoryError, "Out of GPU
# memory ..."). The real backend's free-memory hook is called too, so an
# extension calling a function its package does not have fails here, not on a
# cluster job. That the result does not depend on the chunk is gated elsewhere
# (ka_device_cache_correctness.jl runs a 64-column chunk).
include("ka_backend.jl")
using FastMultipole

const FM = FastMultipole
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

println("Starting KA auto M2L chunk correctness test on $DEV_NAME...")
if !dev_functional()
    println("$DEV_NAME not functional; skipping")
    exit(0)
end

npass = Ref(0); ntot = Ref(0)
function check(label, ok, detail="")
    ntot[] += 1; ok && (npass[] += 1)
    println(label, ": ", ok ? "PASS" : "FAIL", isempty(detail) ? "" : "  " * detail)
end

# a backend whose free memory the test chooses (nothing = no query available)
struct FakeBackend
    free::Union{Nothing,Int}
end
FM._device_free_bytes(b::FakeBackend) = b.free

const P = 6
const TF = Float32
const BASIS = (orders=(P_phi=P, P_active=P),)   # the fields the rule reads
const COL_BYTES = 18 * FM.degree_major_dof(P) * sizeof(TF)   # the rule's per-column estimate
const MIN = ext._KA_MIN_M2L_CHUNK
auto(free) = ext._ka_auto_m2l_chunk(FakeBackend(free), TF, BASIS)

#------- free-memory rule -------#

check("no free-memory query -> cap 2^17", auto(nothing) == 1 << 17, "got $(auto(nothing))")
check("ample memory -> cap 2^17", auto(1 << 40) == 1 << 17, "got $(auto(1 << 40))")
# a tenth of free memory holds exactly 2^14 columns: 2^14 fits, 2^15 does not
let free = 10 * COL_BYTES * (1 << 14)
    check("free fits 2^14 exactly -> 2^14", auto(free) == 1 << 14, "got $(auto(free))")
    check("one byte short -> 2^13", auto(free - 10) == 1 << 13, "got $(auto(free - 10))")
end
check("tiny free memory -> floor $(MIN)", auto(1024) == MIN, "got $(auto(1024))")
check("nothing free -> floor $(MIN)", auto(0) == MIN, "got $(auto(0))")

# the real backend: its extension's query runs and returns a byte count, and the
# cap is per backend (Metal 2^15: 2^16 measured slower there)
let free = FM._device_free_bytes(DEV_BACKEND),
    cap = HAS_METAL ? 1 << 15 : 1 << 17,
    got = ext._ka_auto_m2l_chunk(DEV_BACKEND, TF, BASIS)
    check("$DEV_NAME free-memory query", free isa Integer && free > 0, "free=$(free)")
    check("$DEV_NAME auto chunk in range", MIN <= got <= cap && ispow2(got), "got $got, cap $cap")
end

#------- out-of-memory retry -------#

# throws `err` above `limit` columns, else returns the chunk it was given
oom_above(limit, err) = chunk -> (chunk > limit ? throw(err) : chunk)

check("Metal OutOfMemoryError -> halved to fit",
    ext._ka_with_chunk_halving(oom_above(1 << 14, OutOfMemoryError()), 1 << 17) == 1 << 14)
check("CUDA 'Out of GPU memory' -> halved to fit",
    ext._ka_with_chunk_halving(oom_above(1 << 13,
        ErrorException("Out of GPU memory trying to allocate 1.2 GiB")), 1 << 17) == 1 << 13)
check("fits at once -> unchanged",
    ext._ka_with_chunk_halving(oom_above(1 << 17, OutOfMemoryError()), 1 << 17) == 1 << 17)

# a real device allocation inside the build: the retry returns a usable plan-sized array
let a = ext._ka_with_chunk_halving(c -> (c > 1 << 12 && throw(OutOfMemoryError());
            KernelAbstractions.zeros(DEV_BACKEND, TF, FM.degree_major_dof(P), c)), 1 << 15)
    check("retry hands back the smaller device build", size(a, 2) == 1 << 12, "size $(size(a))")
end

# errors that must not be retried
function throws(f, T)
    try
        f(); return false
    catch err
        return err isa T
    end
end
check("non-memory error propagates at once",
    throws(() -> ext._ka_with_chunk_halving(c -> throw(ArgumentError("bad")), 1 << 17), ArgumentError))
check("still out of memory at the floor -> propagates",
    throws(() -> ext._ka_with_chunk_halving(oom_above(0, OutOfMemoryError()), 1 << 17), OutOfMemoryError))
let calls = Int[]
    ext._ka_with_chunk_halving(c -> (push!(calls, c); c > MIN ? throw(OutOfMemoryError()) : c), 1 << 15)
    check("halves one step at a time to the floor", calls == [1 << 15, 1 << 14, 1 << 13, 1 << 12],
        "calls=$calls")
end

println("\nauto M2L chunk: $(npass[])/$(ntot[]) pass")
npass[] == ntot[] || error("auto M2L chunk gate failed")
