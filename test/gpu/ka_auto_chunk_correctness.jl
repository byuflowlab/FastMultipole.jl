# Gate for the automatic concat-M2L chunk (`ConcatenatedFixedZM2L(0)`,
# ext/ka/ka_workspace_routes.jl): the free-memory rule `_ka_auto_m2l_chunk`.
#
# Every real GPU tried so far has had enough free memory to sit at the cap, so
# the shrinking branches never ran there; here a stand-in backend reports chosen
# free-memory sizes. The real backend's free-memory hook is called too, so an
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

println("\nauto M2L chunk: $(npass[])/$(ntot[]) pass")
npass[] == ntot[] || error("auto M2L chunk gate failed")
