# 052e.0 Tier 0A pre-registration v2 (2026-09-07) — supersedes v1 same date

**Supersedes** `052e0-tier0a-preregistration-2026-09-07.md` after run 1
(log: scratchpad `tier0a_run1.log`; gates snapshot in
`data/052e0-tier0a/gates.txt`). Per v1's failure-handling clause, gates were
not retuned in place; this file records the supersession reason and the two
harness corrections. The kernel under test was NOT modified between runs.

## Run-1 outcome and diagnosis

PASS: A3 (linearity, 0), A5 (slopes −1.9801 both sides), A6 (circulation
error 1e-15, quad delta 9e-12), A7/A9 (kernel vs independent polygon
reference, 1.4e-13). FAIL: A1 (2.000), A2 (0.9501), A8 (flat errors
5.52e-1), A4 (8.8e-12 vs 1e-12).

Diagnosis (verified by hand): the harness's Van Oosterom–Strackee reference
used the sign convention Ω<0 for a viewpoint on the +normal side of a CCW
triangle — opposite to the analytic on-axis branch $\Omega(z)>0$ for $z>0$.
This flipped the identified sign to $s=-1$; A2's residual ≈ $2|\varphi_a|$
(exact sign flip) and A8's mesh-independence are both explained by the
flipped $s$, and A7/A9's 1e-13 agreement shows the kernel and polygon
reference describe the same object. Conclusion available already from run 1:
the implementation realizes $\varphi=-\mu\Omega/4\pi$ (the plan convention)
with jump $[\varphi]=-\mu$ upper-minus-lower. A4's gate was ill-posed:
relative error at far-field points ($|\varphi|\sim6\times10^{-4}$) amplifies
roundoff; the observed 8.8e-12 relative is ~5e-15 absolute.

## Changes relative to v1 (locked before run 2)

1. Reference orientation corrected: $\Omega_{\rm VO}$ is negated so that
   $\Omega>0$ on the +normal (CCW) side, matching the analytic branch. No
   other reference change.
2. A4 becomes absolute: $\max_x |\varphi_{\rm flip}(x)+\varphi(x)| \le
   10^{-12}\,|\mu|$.

## Addendum 2 (2026-09-07, post-review, locked before run 3)

External review of run 2 found two places the harness did not implement the
registered gates literally, plus provenance/scope gaps. Corrections locked
before run 3 (gates themselves unchanged except one added):

1. A6 implements the registered signed comparison $|\Gamma-\mu|/|\mu|$
   (run-2 code used $||\Gamma|-\mu|$, discarding sign; run-2's logged
   $\Gamma=+1.0$ shows the PASS was nevertheless valid). Orientation
   convention now stated: the loop's interior leg ascends $+z$ through the
   disk, positively linking the $+\hat z$ normal, so the expectation is
   $\Gamma=+\mu$.
2. A8 uses the registered metric: refinement of the A2 maximum over ALL
   on-axis points (run-2 code used only $z=0.5R$; reviewer independently
   confirmed $z=0.5R$ is the argmax at every resolution, so run-2 numbers
   are unchanged by this correction).
3. New locked gate **A10 (wrapper parity)**: one panel evaluated through the
   public `pnl.induced(target, body, i)` route must match the low-level
   `pnl._induced` evaluation; gate $\le 10^{-12}|\mu|$. Scope note: Tier 0A
   otherwise exercises the low-level kernel only; the full
   `PanelWake → influence! → direct!` route is exercised in Tiers 0B/1.
4. Provenance: gates.txt additionally records FastMultipole and FLOWPanel
   SHAs with dirty state and tracked-diff hash, thread count, and script
   SHA-256; run logs are preserved under `data/052e0-tier0a/`.

All other fixtures, point sets, gates, and the failure-handling clause carry
over from v1 unchanged, including the addendum's sign/branch identification
protocol. Expected (not gating) run-2 outcomes recorded for honesty:
$s=+1$, $c_\pm=0$, A8 order ≈ 2.
