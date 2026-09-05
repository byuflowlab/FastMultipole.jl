# P5.3 — 052h reverse-leg DEVICE replay (GPU-only; bundle into the next GPU
# job with p34's reverse checks and the device testsets).
#
# Same fixture + host pipeline as p52 (via p52_lib.jl). Runs the device
# reverse pipeline (refresh_cross_producers! build_reverse → reverse Stage B
# vortex B2M + LH M2M → Stage C LH M2L → Stage D LH L2L + L2B → Stage E
# vortex near, singular) and checks:
#   (A) device far field == host far_field(P) to fp-reorder accuracy;
#   (B) device near == host hand near field;
#   (C) device far+near vs production fmm.direct! at P=8 expansion accuracy
#       (p52 observed max relU 1.3e-2 at P=8, q=3, ell=4).
#
# Run: julia --project=<FastMultipole> p53_reverse_device_replay.jl

include(joinpath(@__DIR__, "p52_lib.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
const CUDA = FastMultipole.CUDA
const FM = FastMultipole

const PDEV = 8

# device inputs: targets ("panels") as 3×nt positions; particles as the
# standard ≥7-row source layout (1:3 position, 5:7 strength)
d_tgt = CUDA.CuArray{Float64}(tgt_pos)
pbuf = zeros(8, ns)
pbuf[1:3, :] .= src_pos
pbuf[5:7, :] .= src_str
d_pbuf = CUDA.CuArray{Float64}(pbuf)
d_src = CUDA.CuArray{Float64}(src_pos)

ct = CrossStencilTables(Q, ELL, g.h0, 0.0)
ctx = FM.device_cross_producer_context(ct, SVector{3,Float64}(g.x_min), g.h0,
    nt, ns; build_reverse = true)
FM.refresh_cross_producers!(ctx, d_tgt, d_src)
CUDA.synchronize()
check("no rebuild signal", !ctx.needs_rebuild)
check("reverse lists nonempty", ctx.n_rev_routes > 0 && ctx.n_rev_blocks > 0)

# (A) far field: device Stages B/C/D vs host pipeline
rs = FM.device_cross_reverse_state(ctx, PDEV)
FM.refresh_cross_reverse_multipoles!(rs, ctx, d_pbuf)
FM.refresh_cross_reverse_locals!(rs, ctx)
FM.finish_cross_reverse_locals!(rs, ctx, d_tgt)
CUDA.synchronize()
out_far = Array(rs.d_out)
U_far_dev = [SVector{3,Float64}(out_far[2:4, i]) for i in 1:nt]
U_far_host = far_field(PDEV)
scale_far = maximum(norm.(U_far_host))
dev_far = maximum(norm.(U_far_dev .- U_far_host)) / scale_far
@printf("  device vs host far: max dev %.3e (rel to max |U_far|)\n", dev_far)
check("device far == host far (< 1e-10)", dev_far < 1e-10)

# (B) near field: device Stage E (singular, matching the p52 hand kernel)
FM.apply_cross_reverse_near!(rs, ctx, d_pbuf, d_tgt; reg = false)
CUDA.synchronize()
out_tot = Array(rs.d_out)
U_near_dev = [SVector{3,Float64}(out_tot[2:4, i]) for i in 1:nt] .- U_far_dev
scale_near = max(maximum(norm.(U_near)), eps())
dev_near = maximum(norm.(U_near_dev .- U_near)) / scale_near
@printf("  device vs host near: max dev %.3e (rel to max |U_near|)\n", dev_near)
check("device near == host hand near (< 1e-10)", dev_near < 1e-10)

# (C) total vs the production direct reference
U_tot = U_far_dev .+ U_near_dev
rel = [norm(U_tot[i] - U_ref[i]) / norm(U_ref[i]) for i in 1:nt]
@printf("  device total relU vs production direct: max %.3e  mean %.3e\n",
    maximum(rel), sum(rel) / nt)
check("device total at P=8 expansion accuracy (max relU < 5e-2)",
    maximum(rel) < 5e-2)

@printf("\nP5.3 reverse device replay: %d PASS, %d FAIL\n", npass, nfail)
exit(nfail == 0 ? 0 : 1)
