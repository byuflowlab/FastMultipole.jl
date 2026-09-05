# P3.4 — Stage-A device-producer oracle (052d Step 4, D1 device-native).
#
# Bit-compares the DEVICE cross-pass producer lists (src/cross_stencil_cuda.jl,
# refresh_cross_producers!) against the untouched host CrossStencil prototype
# on the real step-472 snapshot geometry. FP-free integer classification on
# both sides -> EXACT equality is the pass criterion (memo §2).
#
# Checks per config:
#   (1) occupancy parity: per-level cell code sets + per-cell body counts
#       match host LevelCells exactly (both sets);
#   (2) list bit-compare: device far ∪ demoted (as (L, src_code, tgt_code)
#       multisets) == host sweep_config m2l list; device near blocks == host
#       near list;
#   (3) demotion split: every device route/block agrees with the host
#       box-gap rule (strict <, D2);
#   (4) count identity: Σ|A||B| over far + demoted + near == ns · nt.
#
# Configs: the certified operating point (q=12, ell_x=5, Rg=6 mm — expects
# ZERO demotions) plus a fat-guard config (Rg=50 mm) and a small-stencil
# config (q=3, ell_x=4, Rg=20 mm) to exercise the demotion path.
#
# GPU-only: runs in the Step-5 sbatch. Exit code 0 iff all checks pass.
# Run: julia --project=<FastMultipole> p34_stageA_oracle.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
using FastMultipole
using StaticArrays, Printf

const SNAPDIR = get(ENV, "SNAPDIR",
    "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472")
const CONFIGS = ((q = 12, ell_x = 5, Rg = 0.006),
                 (q = 12, ell_x = 5, Rg = 0.050),
                 (q = 3,  ell_x = 4, Rg = 0.020))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
const CUDA = FastMultipole.CUDA

read_i64(f) = reinterpret(Int64, read(joinpath(SNAPDIR, f)))
read_3xn(f) = reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :)
pos = read_3xn("particle_positions_3xN_f64.bin")
verts = read_3xn("panel_vertices_3xM_f64.bin")
conn = read_i64("panel_connectivity_i64.bin"); offs = read_i64("panel_offsets_i64.bin")
panels = let lo = 1
    [begin
         hi = offs[k]
         c = sum(SVector(verts[1, conn[j]], verts[2, conn[j]], verts[3, conn[j]])
                 for j in lo:hi) / (hi - lo + 1)
         lo = hi + 1
         c
     end for k in 1:length(offs)]
end
particles = [SVector(pos[1, i], pos[2, i], pos[3, i]) for i in 1:size(pos, 2)]
ns, nt = length(panels), length(particles)
println("Geometry: step-472 snapshot | ns=$ns panels, nt=$nt particles")

g = CrossGrid(particles)
panel_mat = reduce(hcat, panels)
d_panels = CUDA.CuArray{Float64}(panel_mat)
d_particles = CUDA.CuArray{Float64}(pos)

"Device (level, src node, tgt node) -> (level, src_code, tgt_code) triples."
function canon(levels, sources, targets, src_occ, tgt_occ)
    return [(Int(levels[i]),
             src_occ.node_keys[sources[i]],
             tgt_occ.node_keys[targets[i]]) for i in eachindex(levels)]
end

"Host (si, ti, L) -> (level, src_code, tgt_code) triples."
canon_host(list, src_levels, tgt_levels) =
    [(L, src_levels[L + 1].codes[si], tgt_levels[L + 1].codes[ti])
     for (si, ti, L) in list]

npass = 0; nfail = 0
function check(name, ok)
    global npass, nfail
    ok ? (npass += 1) : (nfail += 1)
    @printf("  %-58s %s\n", name, ok ? "PASS" : "FAIL")
    return ok
end

for cfg in CONFIGS
    println("\n== config q=$(cfg.q) ell_x=$(cfg.ell_x) Rg=$(cfg.Rg) ==")

    # host oracle (untouched prototype; guard applied post-hoc by box gap)
    tq = CrossStencil.UniformQTables(cfg.q)
    src_levels, _, src_exc = build_level_cells(g, panels, cfg.ell_x)
    tgt_levels, _, _ = build_level_cells(g, particles, cfg.ell_x)
    src_exc == 0.0 || error("panel containment failed on host — union box case untested here")
    _, m2l_list, near_list = sweep_config(tq, src_levels, tgt_levels, cfg.ell_x;
        materialize = true)
    demote_host(sc, tc, L) = begin
        w_L = 2 * g.h0 / (1 << L)
        scx, scy, scz = CrossStencil.SharedRadix.morton_decode(sc)
        tcx, tcy, tcz = CrossStencil.SharedRadix.morton_decode(tc)
        o = SVector(Int(tcx) - Int(scx), Int(tcy) - Int(scy), Int(tcz) - Int(scz))
        FastMultipole._cross_demoted(o, w_L, cfg.Rg)
    end
    host_all = canon_host(m2l_list, src_levels, tgt_levels)
    host_far = [t for t in host_all if !demote_host(t[2], t[3], t[1])]
    host_dem = [t for t in host_all if demote_host(t[2], t[3], t[1])]
    host_near = canon_host(near_list, src_levels, tgt_levels)

    # device producer
    ct = CrossStencilTables(cfg.q, cfg.ell_x, g.h0, cfg.Rg)
    ctx = FastMultipole.device_cross_producer_context(ct,
        SVector{3,Float64}(g.x_min), g.h0, ns, nt; build_reverse = true)
    FastMultipole.refresh_cross_producers!(ctx, d_panels, d_particles)
    CUDA.synchronize()
    t_refresh = CUDA.@elapsed begin
        FastMultipole.refresh_cross_producers!(ctx, d_panels, d_particles)
        CUDA.synchronize()
    end
    @printf("  device refresh (steady-state rebuild): %.4f s\n", t_refresh)
    lists = FastMultipole.download_cross_lists(ctx)
    check("root box preserved (no rebuild signal)", !lists.needs_rebuild)

    # (1) occupancy parity, both sets
    for (name, occ, hl) in (("panel", lists.panels, src_levels),
                            ("particle", lists.particles, tgt_levels))
        ok = true
        for L in 0:cfg.ell_x
            r = occ.level_offsets[L + 1] + 1:occ.level_offsets[L + 2]
            dev_codes = occ.node_keys[r]
            dev_counts = occ.node_ranges[2, r]
            hc = hl[L + 1]
            ok &= dev_codes == hc.codes
            ok &= dev_counts == Int32[CrossStencil.cellcount(hc, i)
                                      for i in 1:CrossStencil.ncells(hc)]
        end
        check("$name occupancy: codes + per-cell counts (all levels)", ok)
    end

    # (2) list bit-compare
    r = lists.routes; b = lists.blocks
    dev_far = canon(r.levels, r.sources, r.targets, lists.panels, lists.particles)
    dev_dem = canon(b.levels[1:b.n_demoted], b.sources[1:b.n_demoted],
        b.targets[1:b.n_demoted], lists.panels, lists.particles)
    dev_near = canon(b.levels[b.n_demoted + 1:end], b.sources[b.n_demoted + 1:end],
        b.targets[b.n_demoted + 1:end], lists.panels, lists.particles)
    check("far routes == host far (exact multiset, n=$(length(dev_far)))",
        sort(dev_far) == sort(host_far))
    check("demoted blocks == host demoted (n=$(length(dev_dem)))",
        sort(dev_dem) == sort(host_dem))
    check("near blocks == host near (n=$(length(dev_near)))",
        sort(dev_near) == sort(host_near))

    # (3) demotion split honors the strict-< gap rule on the device offsets
    ok3 = all(!FastMultipole._cross_demoted(SVector{3,Int}(r.offsets[:, i]),
            2 * g.h0 / (1 << r.levels[i]), cfg.Rg) for i in 1:length(r.levels))
    ok3 &= all(FastMultipole._cross_demoted(SVector{3,Int}(b.offsets[:, i]),
            2 * g.h0 / (1 << b.levels[i]), cfg.Rg) for i in 1:b.n_demoted)
    check("route/block offsets consistent with gap rule", ok3)

    # (4) count identity over the full partition
    total = Int128(0)
    cnt(occ, node) = Int128(occ.node_ranges[2, node])
    for i in 1:length(r.levels)
        total += cnt(lists.panels, r.sources[i]) * cnt(lists.particles, r.targets[i])
    end
    for i in 1:length(b.levels)
        total += cnt(lists.panels, b.sources[i]) * cnt(lists.particles, b.targets[i])
    end
    check("count identity Σ|A||B| == ns·nt ($total)", total == Int128(ns) * Int128(nt))

    # (5) 052h REVERSE leg (particles→panels): same checks against the host
    # sweep with the roles swapped — particles are the explicit sources,
    # panels the dense-occupancy targets.
    _, rev_m2l_list, rev_near_list = sweep_config(tq, tgt_levels, src_levels,
        cfg.ell_x; materialize = true)
    rhost_all = canon_host(rev_m2l_list, tgt_levels, src_levels)
    rhost_far = [t for t in rhost_all if !demote_host(t[2], t[3], t[1])]
    rhost_dem = [t for t in rhost_all if demote_host(t[2], t[3], t[1])]
    rhost_near = canon_host(rev_near_list, tgt_levels, src_levels)
    rr = lists.rev_routes; rb = lists.rev_blocks
    rdev_far = canon(rr.levels, rr.sources, rr.targets, lists.particles, lists.panels)
    rdev_dem = canon(rb.levels[1:rb.n_demoted], rb.sources[1:rb.n_demoted],
        rb.targets[1:rb.n_demoted], lists.particles, lists.panels)
    rdev_near = canon(rb.levels[rb.n_demoted + 1:end], rb.sources[rb.n_demoted + 1:end],
        rb.targets[rb.n_demoted + 1:end], lists.particles, lists.panels)
    check("REV far routes == host far (n=$(length(rdev_far)))",
        sort(rdev_far) == sort(rhost_far))
    check("REV demoted blocks == host demoted (n=$(length(rdev_dem)))",
        sort(rdev_dem) == sort(rhost_dem))
    check("REV near blocks == host near (n=$(length(rdev_near)))",
        sort(rdev_near) == sort(rhost_near))
    rtotal = Int128(0)
    for i in 1:length(rr.levels)
        rtotal += cnt(lists.particles, rr.sources[i]) * cnt(lists.panels, rr.targets[i])
    end
    for i in 1:length(rb.levels)
        rtotal += cnt(lists.particles, rb.sources[i]) * cnt(lists.panels, rb.targets[i])
    end
    check("REV count identity Σ|A||B| == nt·ns ($rtotal)",
        rtotal == Int128(nt) * Int128(ns))
end

@printf("\nP3.4 Stage-A oracle: %d PASS, %d FAIL\n", npass, nfail)
exit(nfail == 0 ? 0 : 1)
