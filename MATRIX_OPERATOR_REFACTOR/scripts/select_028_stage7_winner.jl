# Print "<policy> <q2-q3-q4-q5>" for the newest complete Stage 7 frontier CSV.
using DelimitedFiles

const DIR = normpath(joinpath(@__DIR__, "..", "data", "feasibility_1m_10ms"))
const FRONTIER = Set(["hier5", "sched6-5-5-5", "sched6-6-5-5",
    "sched6-6-6-5", "hier6"])
const GATE = 1.19e-3

files = sort(filter(f -> startswith(basename(f), "cuda_") && endswith(f, ".csv") &&
    !endswith(f, ".classes.csv"), readdir(DIR; join=true));
    by=mtime, rev=true)
for file in files
    lines = readlines(file)
    isempty(lines) && continue
    header = split(first(lines), ',')
    ip = findfirst(==("policy"), header)
    ifit = findfirst(==("fit"), header)
    it = findfirst(==("verdict_step_ms"), header)
    ie = findfirst(==("err_gradient_rel_rms"), header)
    any(isnothing, (ip, ifit, it, ie)) && continue
    rows = [split(line, ',') for line in lines[2:end]]
    policies = Set(row[ip] for row in rows if length(row) >= length(header))
    FRONTIER ⊆ policies || continue
    admissible = filter(rows) do row
        row[ip] in FRONTIER && row[ifit] == "true" &&
            tryparse(Float64, row[ie]) !== nothing && parse(Float64, row[ie]) <= GATE
    end
    isempty(admissible) && error("newest complete Stage 7 frontier has no admissible row: $file")
    winner = first(sort(admissible; by=row -> parse(Float64, row[it])))
    policy = winner[ip]
    schedule = policy == "hier5" ? "5-5-5-5" :
        policy == "hier6" ? "6-6-6-6" : policy[6:end]
    println(policy, ' ', schedule)
    exit()
end
error("no complete Stage 7 frontier CSV found in $DIR")
