# One-process runner for the ka_*_correctness.jl suites.
#
# Each suite is included into its own fresh module inside THIS Julia session,
# so package load and Metal kernel compilation are paid once instead of once
# per suite (the per-process floor was ~15 s even for a trivial gate). Output
# per suite goes to logs/<suite>.log; one summary line per suite is printed,
# matching run_suites.sh, and the exit status is the number of failing suites.
#
#   julia --project=. run_suites.jl              # all suites
#   julia --project=. run_suites.jl m2m l2l      # suites matching substrings
#
# Suite conventions this relies on: a suite calls `exit(0)` to skip when the
# device is missing (shadowed here to raise a Skip), throws (error() or a
# failing @test) on failure, and prints its own summary line on success.

using Printf

const FAIL_RE = r"FAIL|THREW|Test Failed|ERROR:|✗|MethodError|not functional"

struct SuiteSkip <: Exception
    code::Int
end

function run_suite(path::String, log::String)
    m = Module(Symbol("Suite_", replace(basename(path), "." => "_")))
    # a suite's `exit(0)` means "skip"; anything else is a failure
    Core.eval(m, :(exit(code=0) = throw($SuiteSkip(code))))
    # a bare module has no Main-style include/eval
    Core.eval(m, :(include(p) = Base.include($m, p)))
    Core.eval(m, :(eval(x) = Core.eval($m, x)))
    status = :pass
    open(log, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                try
                    Base.include(m, path)
                catch err
                    if err isa SuiteSkip && err.code == 0
                        status = :skip
                    else
                        status = :fail
                        showerror(io, err, catch_backtrace())
                        println(io)
                    end
                end
            end
        end
    end
    return status
end

cd(@__DIR__)
mkpath("logs")
suites = filter(f -> occursin(r"^ka_.*_correctness\.jl$", f), readdir())
if !isempty(ARGS)
    suites = filter(f -> any(occursin(pat, f) for pat in ARGS), suites)
end
isempty(suites) && (println("no suites matched: ", join(ARGS, " ")); exit(0))

nfail = 0
for f in suites
    name = f[1:end-3]
    log = joinpath("logs", name * ".log")
    t0 = time()
    status = run_suite(joinpath(@__DIR__, f), log)
    dt = round(Int, time() - t0)
    text = read(log, String)
    if status == :pass && !occursin(FAIL_RE, text)
        ms = collect(eachmatch(r"[0-9]+/[0-9]+ pass|All .* passed|gate passed", text))
        @printf("%-42s PASS  %3ds  %s\n", name, dt, isempty(ms) ? "" : ms[end].match)
    elseif status == :skip
        @printf("%-42s SKIP  %3ds\n", name, dt)
    else
        global nfail += 1
        @printf("%-42s FAIL  %3ds  -> %s\n", name, dt, log)
        for (i, line) in enumerate(split(text, '\n'))
            occursin(FAIL_RE, line) && println("    ", i, ": ", line)
            i > 2000 && break
        end
    end
    flush(stdout)
end

println("---")
nfail == 0 ? println("all $(length(suites)) suites PASS") :
             println("$nfail of $(length(suites)) suites FAILED (full output in logs/)")
exit(nfail)
