# Adaptive Radix Octree (2:1-Balanced) — Task 038

## 0. Scope, evidence, and reused conventions

This artifact derives the complete theory for a 2:1-balanced adaptive Morton
octree on the radix path, preserving the batched, translation-invariant,
device-resident operator machinery. It covers: tree definition and
construction as sort/scan/compact (§1); U/V/W/X interaction lists with
generation algorithms (§2); an exact-once coverage proof at both supported
near radii (§3); M2T and S2L operators with constant-$P$-consistent error
bounds and Lamb–Helmholtz coverage (§4); the per-cell $\sigma$ geometry gate
replacing the global $\sigma_{\max}$ form (§5); the cost and capacity model
including a synthetic multi-scale case (§6); and the refresh/rebuild policy
(§7). Verification artifacts are listed in §8.

**Entry-gate evidence (recorded per the 038 task file).** The `037b`
rotor-wake campaign measured multi-scale density as a binding cost of the
uniform-depth grid: the auto-selected uniform geometry costs 7.16x at
$n=10^6$ (238.7 → 33.3 ms at pinned $\ell=8$) and 1.81x at $n=10^5$; the
mechanism is occupancy contrast (rotor max/mean bodies per occupied cell
5–7× at every level vs wake ≤1.9×; the top-1% densest cells hold ~6% of all
bodies), and the uniform path's $\ell\le8$ cap plausibly still binds at
$n=10^6$ (nearfield ~70% of the $\ell=8$ evaluation). Notably, the
$\sigma_{\max}$ *geometry-gate* mechanism does **not** bind on the rotor
(thin young tip cores keep $\sigma_{\max}$ small); the per-cell $\sigma$
gate of §5 is therefore derived as a correctness/generality requirement
(`CoreSpreading`-grown $\sigma$), while the measured win mechanism this
design targets is density contrast.

**Reused conventions (unchanged).**

- `025` (`theory/hierarchical-rigid-m2l-stencil.md`): near family
  $N_q=\{o\in\mathbb Z^3:\|o\|^2\le q\}$ with the two required members
  $q=3$ (classic, $|N_3|=27$) and $q=12$ ($\theta=0.5$, $|N_{12}|=179$);
  the source-major phase table; the level-scaling law
  $K(sr)=s^{-1}\Lambda(s)K(r)\Lambda(s)$ (and its Lamb–Helmholtz form with
  distinct $\Lambda_t,\Lambda_s$) that lets one offset-indexed operator
  table serve every level. The V-list path below reuses these operator
  tables **unchanged** (§2.4).
- `008d` (`theory/constant-p-error-stencil.md`): the conservative
  constant-$P$ bound
  $B(P,d,A)=\frac{2A}{\rho(c-2)}\left(\frac{1}{c-1}\right)^{P+1}$ with
  $c=2\|d\|/\sqrt3$, valid for $c>2$.
- `008h` (`theory/lamb-helmholtz-accuracy-order.md`): the channel order
  rule $P_\chi=P_\phi+1$, $P_{\rm active}=P_\chi$, and the LH stencil tails
  $B_{\rm LH}=B(P_\phi,d,A_\phi)+(1+2R)\,B(P_\phi+1,d,A_\chi)$.
- `031a` (`theory/kernel-splitting-nearfield.md`): the regularized-kernel
  contract — every pair with $r/\sigma_s\le\rho_t$ must be evaluated by the
  regularized direct kernel ($\rho_t=4.789$ at the phase gate
  $\varepsilon=10^{-3}$, overlap $\beta=2$); beyond $\rho_t$ the singular
  kernel's omitted tail is bounded and inside the budget.
- `007` (`theory/coefficient-buffer-layout.md`): compressed complex basis,
  $0\le m\le n\le P$, `harmonic_index(n,m) = n(n+1)/2 + m + 1`, two real
  lanes, channels $(\phi,\chi)$ for `Val(true)`.
- `008f` (`theory/radix-sort-clustering.md`): root cube, quantization,
  Morton keys, stable sort, occupied-cell compaction — all reused verbatim
  at the deepest level.

Scoped derivation row: `theory/`, `scripts/`, `data/` only; no production
`src/` changes; does not reopen the Theory Phase hard gate.

## 1. Tree definition and construction

### 1.1 Occupied linear octree

Fix the root cube (center $c_0$, half-width $h_0$, per `008f`), a split
threshold $K_{\max}\ge1$, and a depth cap $\ell_{\max}$. A *cell* at level
$\ell$ is an integer triple $c\in[0,2^\ell)^3$ with width

$$
\Delta_\ell = \frac{2h_0}{2^\ell},
$$

occupying the half-open box $x_{\min}+\Delta_\ell\,[c, c+1)$ per axis (upper
boundary clamped per `008f`). The **occupied adaptive tree** is defined by
the recursive rule: the root (level 0) is a node; a node is **split** into
its *occupied* children (the up-to-8 level-$(\ell{+}1)$ sub-cells containing
at least one body) iff its population exceeds $K_{\max}$ and $\ell <
\ell_{\max}$; otherwise it is a **leaf**. Every node is occupied; empty
sub-cells are never materialized (occupancy compaction, matching the radix
path's ethos). Consequences:

- occupied leaves have pairwise-disjoint boxes and contain every body;
- every leaf has population $\le K_{\max}$ **except** leaves at
  $\ell_{\max}$ (the depth cap) and leaves produced by the balance sweep
  (§1.4), which may be split below $K_{\max}$;
- every internal node produced by a *population* split has population
  $> K_{\max}$.

The nodes at each level form the per-level node sets over which the
existing hierarchical machinery (per-level M2M/M2L/L2L, level-scaled
operator tables, occupancy-epoch caches) operates; the only structural
novelty is that *leaves now appear at multiple levels*.

### 1.2 Construction from the sorted body stream

Quantize every body once at the deepest level $\ell_{\max}$ and form
full-depth Morton keys ($3\ell_{\max}$ bits, `008f` bit interleave). Sort
bodies by full-depth key (stable). The **prefix property** — the level-$\ell$
key of a body is the leading $3\ell$ bits of its full-depth key, and Morton
order sorts by every prefix simultaneously — gives:

> For every cell at every level, the bodies it contains form one contiguous
> range of the sorted order.

Construction is then a top-down frontier sweep. Maintain a frontier of
(cell, body-range) records, initialized to the root:

1. **Populations**: range length per frontier cell (already explicit).
2. **Split flags**: `pop > K_max && level < ell_max` (§5.4 optionally adds
   a per-cell $\sigma$ split veto here).
3. **Leaf compaction**: cells with clear flags are appended to the node
   table as leaves (stream compaction).
4. **Child expansion**: for each split cell, its children are the distinct
   values of the next 3 key bits within its range; the child sub-ranges are
   the runs of those bits. Run boundaries are found by flagging positions
   where the 3-bit value changes (a boundary flag over the range) followed
   by a scan/compact; equivalently, 8 binary searches per split cell.
5. The children become the next frontier; iterate to at most
   $\ell_{\max}$ rounds.

Every step is a map (flags), scan (offsets/populations), compact
(leaf/child emission), or binary search over sorted keys — the primitives
already present in the device radix path (`020a` device sort, flag/scan/
compact route generation in `026`/`027`). This is the device-feasibility
argument required by the task; device code itself is row `041`.

The node table records, per node: level, integer coordinate (equivalently
the level-truncated key), contiguous sorted-body range (the *subtree*
range — internal nodes keep their full range), parent index, child index
range, and leaf flag. A per-level index (the `levels_index` analog) is
produced by a stable sort of nodes by level, or directly by emitting one
level per frontier round.

### 1.3 Body permutation and buffers

The sorted permutation and inverse permutation follow `008f` unchanged.
All per-body buffers are ordered by the single full-depth sort; no
per-level re-sorts occur. Expansion buffers are allocated per node (leaf
*and* internal), indexed by node id, sized by the `007` layout at
$P_{\rm active}$.

### 1.4 2:1 balance

**Definition.** Two cells *touch* if their closed boxes intersect
(equivalently: with $A$ the coarser, the per-axis clamp distance of $B$'s
coordinate to $A$'s tile interval at $B$'s level — see §2.2 — is at most 1
on every axis). The occupied leaf set is **2:1 balanced** iff every pair of
touching occupied leaves differs by at most one level.

**Sweep.** Iterate to a fixed point: whenever occupied leaves $A$ (level
$\ell$) and $B$ (level $\ge\ell+2$) touch, split $A$ (into its occupied
children; bodies re-bucket by the presorted keys). In sort/scan/compact
form (the standard bottom-up balance construction, Sundar-style): process
levels deepest-first; each leaf at level $\ell$ emits the keys of its
$\le26$ parent-level ($\ell-1$) touching neighbor cells; sort and unique
the emitted keys; match them against the coarse-leaf key intervals
(merge/binary search); flag and split matched leaves; repeat. At most
$\ell_{\max}$ rounds terminate, because splits only deepen leaves and
levels are capped.

**Termination and feasibility.** A balance split always succeeds: the split
leaf is occupied, so it has $\ge1$ occupied child; the depth cap never
obstructs balance because a violation at $(\ell,\ell')$ with
$\ell'\ge\ell+2$ requires only refinement to $\ell'-1\le\ell_{\max}-1$.

**Role.** Balance is *not* required for correctness — the exact-once proof
of §3 holds for any occupied leaf partition, and the verification script
checks unbalanced trees too. Balance is a performance/regularity device: it
bounds the level spread of touching U-list partners to $\pm1$, keeps W/X
lists shallow in practice (§2.5), and bounds the per-leaf list sizes used
by the capacity model (§6).

## 2. Interaction lists

### 2.1 Near predicates

Let $q_\ell$ be the near radius at level $\ell$ (the `025` policy carries a
per-level radius vector `level_radii2`; a constant $q\in\{3,12\}$ is the
common case). For same-level cells with offset $o = c_T - c_S$ (repo
convention $o = T - S$):

$$
\mathrm{near}_\ell(A,B) \iff o \in N_{q_\ell} \iff \|o\|^2 \le q_\ell .
$$

For mixed-level cells — $A$ at $\ell_A$, $B$ at $\ell_B$, say
$\ell_A<\ell_B$ ($A$ coarser) — define the near test **on the lattice of
the finer cell**. $A$ covers the tile interval
$[c_A 2^{k}, (c_A{+}1)2^{k}-1]$ per axis at level $\ell_B$, where
$k=\ell_B-\ell_A$. The minimal tile offset is obtained by per-axis
clamping:

$$
d_i \;=\;
\begin{cases}
c_{B,i} - c_{A,i}2^{k}, & c_{B,i} < c_{A,i}2^{k},\\
0, & c_{B,i} \in [c_{A,i}2^{k},\,(c_{A,i}{+}1)2^{k}-1],\\
c_{B,i} - \bigl((c_{A,i}{+}1)2^{k}-1\bigr), & \text{otherwise},
\end{cases}
$$

and

$$
\mathrm{near}(A,B) \iff \|d\|^2 \le q_{\ell_B}.
$$

That is: $B$ is near $A$ iff some level-$\ell_B$ tile of $A$ lies in $B$'s
near set. Three properties motivate this choice:

1. it reduces to the same-level rule when $\ell_A=\ell_B$ ($k=0$, the tile
   interval is one cell);
2. it is integer-exact and $O(1)$ per pair (three clamps);
3. **far at the finer scale** means *every* point of the coarse cell lies
   in a level-$\ell_B$ tile that is separated from the finer cell in the
   `025` sense — which is exactly the geometry under which the finer cell's
   expansion (the one W/X use, §4) satisfies the constant-$P$ error bound.

For $q=3$, mixed-level near coincides with touching ($\|d\|^2\le3
\iff \|d\|_\infty\le1$).

### 2.2 Dual-tree recursion (DTR)

Lists are defined by a recursion over ordered (target, source) node pairs,
started at (root, root):

```text
DTR(A, B):
  if far(A, B):                                  # not near (§2.1), §5 gate passes
    if level(A) == level(B):   emit V(A, B)      # M2L, offset class
    elseif level(A) < level(B): emit W(A, B)      # A is a leaf (invariant): M2T
    else:                       emit X(A, B)      # B is a leaf (invariant): S2L
  else:                                          # near
    if leaf(A) and leaf(B):    emit U(A, B)      # direct (includes A == B)
    elseif level(A) == level(B):
      if leaf(A):     recurse (A, child) for children of B
      elseif leaf(B): recurse (child, B) for children of A
      else:           recurse (childA, childB) for all child pairs
    elseif level(A) < level(B): recurse (A, child) for children of B   # A leaf
    else:                       recurse (child, B) for children of A   # B leaf
```

**Invariant 1 (mixed-level pairs).** In every reached mixed-level pair the
coarser member is a leaf. *Proof:* the root pair is same-level; same-level
recursion creates a mixed pair only by splitting the internal member
against a same-level leaf; mixed-level recursion splits only the finer
member, preserving the coarser-is-leaf property. $\square$

**Invariant 2 (parent nearness).** Every recursion step descends only near
pairs, and the root pair is near ($o=0$). Hence every emitted V pair has
near parents at the parent level's radius, and every emitted W/X pair's
parent pair (the pair from which it was reached) was near. $\square$

### 2.3 The four lists

- **U (direct).** Ordered pairs of near occupied leaves, including the
  self pair. Evaluated by the existing direct/nearfield kernels
  (regularized per the `031a` contract; §5 guarantees the cutoff coverage).
  With $q=3$ and 2:1 balance, touching partners differ by at most one
  level; with $q=12$, non-touching near partners may differ by more (the
  capacity model §6.2 accounts for this).
- **V (M2L).** Same-level far pairs with near parents — see §2.4.
- **W (M2T).** Pairs (coarse target leaf $A$, finer source cell $B$), far
  at $B$'s level: the multipole expansion of $B$ (which summarizes $B$'s
  whole subtree) is evaluated directly at $A$'s bodies (§4.1).
- **X (S2L).** Pairs (finer target cell $A$, coarse source leaf $B$), far
  at $A$'s level: $B$'s bodies are accumulated directly into $A$'s local
  expansion (§4.2), which the ordinary L2L/L2B downward pass then carries
  to $A$'s subtree.

W and X are exact duals: exchanging target and source roles maps one to
the other ($N_q=-N_q$ makes the near predicate symmetric, matching the
`025` push/pull bijection).

### 2.4 V-classes: `025` operator tables reused unchanged

Every emitted V pair $(A,B)$ at level $\ell$ satisfies, by Invariant 2 and
the far test:

$$
o = c_A - c_B \notin N_{q_\ell},
\qquad
p = \operatorname{fld}(u+o,2) \in N_{q_{\ell-1}},
\quad u = c_B \bmod 2 \ \text{(source phase)},
$$

(Orientation note, corrected on the `039` touch per the `038` re-approval
review: the phase identity holds for $o = c_T - c_S = c_A - c_B$ paired
with the *source* phase $u = c_B \bmod 2$ — writing $c_B = 2p_B + u$ gives
$p_A - p_B = \operatorname{fld}(c_B + o, 2) - p_B = \operatorname{fld}(u+o,2)$
— matching the `025` definition $o = T - S$. Set-level membership is
orientation-independent because $N_q = -N_q$ and the phase-table family is
closed under the push/pull bijection, so no downstream claim changes.)

which is precisely the `025` source-major phase-table membership
$V_{\rm push}(u)$. Therefore:

- the admissible offset set at each level is a subset of the finite `025`
  phase-table set (Chebyshev reach $\le3$ for $q=3$, $\le7$ for $q=12$;
  per-phase cardinalities 189 / 1253). This holds **with the §5 gate
  active** because demotion is sticky (§5.2): V is emitted only on
  never-demoted paths, where descent used the geometric near predicate,
  so Invariant 2's parent nearness is geometric — verified computationally
  on both gated and ungated lists;
- batching by $(\ell, o)$ class and the level-scaling law
  $K(sr)=s^{-1}\Lambda(s)K(r)\Lambda(s)$ (scalar and LH forms) apply
  verbatim — **no new operator tables** are required, satisfying the task
  constraint. The V-list path is byte-identical in operator content to the
  uniform hierarchical path; only *which* $(\ell,o,A,B)$ tuples appear
  changes.
- coarse levels need no special branch: the `025` argument (level 1 has no
  separated pair; every separated level-2 pair has a near parent) holds
  unchanged because it is a statement about the lattice, not occupancy.

### 2.5 W/X level structure

**Proposition (classic complete-tree limit).** In a *complete* (unpruned)
2:1-balanced octree with $q=3$, every W (X) partner is exactly one level
finer than its leaf. *Proof sketch:* a W partner at $\ell_B\ge\ell_A+2$
requires an internal node $C$ at $\ell_A+1$ touching the leaf $A$ (the
recursion reaches $B$ only through near = touching ancestors). In a
complete tree, at least one geometric child of $C$ touches $A$; if it is a
leaf, 2:1 balance is violated ($\ell_A+2$ vs $\ell_A$); if internal,
recurse — an infinite descent, impossible. Hence no such $C$ exists.
$\square$

**Occupancy-pruned caveat.** With occupied-only children, the geometric
child of $C$ that touches $A$ may be empty and unmaterialized, while an
occupied non-touching child at $\ell_A+2$ is far and becomes a W entry two
(or more) levels finer, without violating leaf balance. This is correct
(the exact-once proof and error bounds hold at any level difference — the
far test is applied at the *finer* level, where the expansion lives), and
it is the honest statement for this design: **W/X entries are one level
finer in the classic complete-tree limit; under occupancy pruning and/or
$q=12$ they may be deeper, bounded only by $\ell_{\max}$.** The
verification script measures the level-difference distribution (it is
overwhelmingly concentrated at one level on all test cases), and the
capacity model bounds W/X sizes without a depth assumption (§6.2).

### 2.6 Pipeline placement

The full pass over the adaptive tree is the existing resident pipeline
plus two body-mediated stages:

1. **B2M** at every leaf (leaf level varies; the existing kernels are
   per-node and level-agnostic).
2. **M2M** upward over per-level node sets (existing level-scaled
   operators; a parent gathers only occupied children).
3. **Horizontal**: V-list M2L by $(\ell,o)$ class (existing strategies and
   tables, §2.4); **X-list S2L** into local expansions (§4.2). Both write
   local expansions, so they precede the downward pass.
4. **L2L** downward over per-level node sets (existing operators); a leaf
   at a coarse level simply terminates its branch.
5. **L2B** at every leaf, plus **W-list M2T** directly to leaf bodies
   (§4.1), plus **U-list direct** (existing nearfield kernels). These
   three accumulate into the same per-body outputs, matching the current
   nearfield/far-field accumulation contract.

### 2.7 List generation in flag/scan/compact form

The DTR is realized on device as a frontier-based pair sweep (the standard
GPU dual-tree formulation), mirroring the construction sweep:

1. Frontier = array of (A, B) node-id pairs; initialize to [(root, root)].
2. **Classify** each pair in $O(1)$: evaluate near (three clamps), the §5
   gate (one compare against the precomputed per-node
   $\rho_t\sigma_{\max}$, §5.2, plus the sticky lineage bit carried in the
   pair record), leaf flags, and levels → one of {U, V, W, X, expand-A,
   expand-B, expand-both}.
3. **Emit**: scan the per-class counts; compact U/V/W/X records into their
   output arrays.
4. **Expand**: scan the child-pair counts (product of child counts for
   expand-both); compact the next frontier.
5. Iterate. Each round strictly increases $\min(\ell_A)+\min(\ell_B)$ over
   the frontier, so at most $2\ell_{\max}$ rounds run.

V records carry $(\ell, o, A, B)$; a final sort by $(\ell,o)$ key plus
boundary flag/scan/compact yields the class-partitioned layout the
existing resident M2L strategies consume (the same compaction the `026`/
`027` windowed generation performs). U/W/X records are grouped by target
leaf (segmented sort by target id) for coalesced per-leaf kernels.

An equivalent per-target closed-form generation (colleague walk) exists
for U/W at $q=3$ and is noted for `039` as a CPU reference, but the
frontier sweep is the primary form because it needs no per-level neighbor
tables and expresses the $\sigma$ gate naturally.

## 3. Exact-once coverage

### 3.1 Bucket coverage semantics

For a bucket $E(A,B)$ let $\mathrm{bodies}(A)$ be the bodies of $A$'s cell
range (for a leaf this is its own range) and $\mathrm{sub}(A)$ the bodies
of $A$'s subtree (its contiguous range; for a leaf,
$\mathrm{sub}(A)=\mathrm{bodies}(A)$). The ordered body pairs covered are:

$$
\begin{aligned}
U(A,B)&:\ \mathrm{bodies}(A)\times\mathrm{bodies}(B) && \text{(direct)}\\
V(A,B)&:\ \mathrm{sub}(A)\times\mathrm{sub}(B) && \text{(M2M} \to \text{M2L} \to \text{L2L/L2B)}\\
W(A,B)&:\ \mathrm{bodies}(A)\times\mathrm{sub}(B) && \text{(M2M} \to \text{M2T)}\\
X(A,B)&:\ \mathrm{sub}(A)\times\mathrm{bodies}(B) && \text{(S2L} \to \text{L2L/L2B)}
\end{aligned}
$$

Since $A$ is a leaf in W and $B$ is a leaf in X, all four are instances of
$\mathrm{sub}(A)\times\mathrm{sub}(B)$.

### 3.2 Theorem (exact-once)

**Theorem.** For any occupied leaf partition (balanced or not), any
per-level near radii, and any admissible $\sigma$-gate predicate (§5), the
DTR emits buckets whose covered ordered body-pair sets partition
$\mathrm{bodies}\times\mathrm{bodies}$ — every ordered pair, including
same-cell and same-body pairs, is covered exactly once.

*Proof.* Consider the recursion tree rooted at (root, root).

1. *Partition step.* When DTR expands a pair $(A,B)$, the child pairs'
   covered sets partition $\mathrm{sub}(A)\times\mathrm{sub}(B)$: the
   occupied children of a node partition its bodies (every body lies in
   exactly one occupied child), and expanding one or both sides takes the
   corresponding product partition.
2. *Termination.* Each expansion strictly increases $\ell_A+\ell_B$, which
   is bounded by $2\ell_{\max}$; and a pair that cannot expand (both
   leaves) is always emitted (far ⇒ V/W/X by level comparison; near ⇒ U).
   Hence every recursion branch terminates in an emission.
3. *Disjointness and coverage.* By induction on the recursion tree: the
   covered set of a node equals the disjoint union of its children's
   covered sets (step 1), and terminal nodes are emitted exactly once.
   The root covers all ordered pairs. $\square$

Note what the proof does *not* use: the specific form of the near
predicate — any per-pair classification rule, including one carrying
recursion-path state like the §5 sticky-demotion lineage bit, yields a
valid partition, which is what makes the §5 gate free — nor balance, nor
radius constancy across levels. Those choices govern *efficiency* and
*error*, not coverage.

Self-interaction: the pair $(A,A)$ is near ($o=0$) at every level, so it
descends to leaf self pairs emitted in U; the direct kernel's existing
self-pair handling (self-body exclusion / self-influence policy) is
unchanged.

Finite-domain truncation drops nonexistent partners exactly as in `025`:
a missing source cell removes a pair, never duplicates one.

### 3.3 Uniform-limit parity with `025`

**Proposition.** If every occupied leaf is at one depth $\ell$ (the
uniform grid), the DTR emits exactly the `025` construction: U = the
near leaf pairs; V = each separated leaf pair at its first separated
ancestor level $L_*=\min\{L\ge2 : o_L\notin N_{q_L}\}$; W and X are empty.

*Proof.* Mixed pairs never arise (no leaf exists above $\ell$, so
same-level near pairs always split both sides), so W/X are empty. A leaf
pair with near leaf offset descends to leaf level and lands in U. A
separated leaf pair's ancestor chain is near at levels $<L_*$ (descent
continues) and far first at $L_*$ (emit V there); this is exactly the
`025` first-separated-ancestor rule, whose downward monotonicity argument
is not even needed here because the recursion stops at the first far
level by construction. $\square$

This is the uniform-limit parity gate row `039` implements against.

### 3.4 Computational verification

`scripts/adaptive_octree_verify.jl` (stdlib-only) verifies on five
distributions (uniform, multi-scale at 30× and 100× contrast, rotor-like
filament, adversarial two-cluster), for $q\in\{3,12\}$,
$K_{\max}\in\{16,64\}$, balanced and unbalanced:

- exact-once coverage by brute-force pair painting (assert count $\equiv1$
  over all $n^2$ ordered pairs);
- the 2:1 balance fixed point and property;
- V-class admissibility per §2.4 (separated offset, near parent, Chebyshev
  reach);
- uniform-limit parity per §3.3 against an independent $L_*$
  implementation;
- the §5 gate contract and coverage under demotion;
- W/X level-difference distributions (§2.5).

Evidence tables land in `data/adaptive_octree/`.

## 4. M2T and S2L operators

### 4.1 M2T (multipole-to-target)

Conventions: Gumerov-normalized regular and irregular solid harmonics (the
`full-m2l-composition.md` / `src/harmonics.jl` forms),

$$
R_n^m(x) = \frac{(-1)^n\, i^{|m|}\, \rho^{n}\, P_n^{|m|}(\cos\theta)\,
e^{im\varphi}}{(n+|m|)!},
\qquad
S_n^m(x) = (-1)^m\, i^{|m|}\, \rho^{-n-1}\, P_n^{|m|}(\cos\theta)\,
e^{im\varphi}\,(n-|m|)! ,
$$

compressed complex storage $0\le m\le n$ with the real-lane contraction
convention of `real-solid-harmonic-transforms.md` (the $m>0$ factor 2
carries the conjugate $-m$ terms).

**Definition.** For a W pair (target leaf $A$, source cell $B$ with
multipole $M$ about $c_B$), M2T evaluates the truncated multipole field at
each target body $x_t\in A$:

$$
u(x_t) \;=\; \sum_{n=0}^{P}\sum_{m=0}^{n} s_m\,
\Re\!\left[ S_n^m(x_t-c_B)\, M_n^m \right],
\qquad s_m = \begin{cases}1,&m=0\\2,&m>0\end{cases}
$$

with gradient and Jacobian obtained by the same degree-shift structure as
the local evaluation (`008e`): the $\phi$ contribution to the field at
degree $n$ reads $S_{n+1}^{m}$ (irregular harmonics one degree up, indices
$m-1,m,m+1$ with the same combination coefficients as the local $G$
operator), and the Hessian/Jacobian by a second application of the
$\chi$-free structure. This functional already exists in-repo as
`evaluate_multipole` (`test/evaluate_multipole.jl`, included by
`src/error.jl`), including the Lamb–Helmholtz channel; row `040`
productionizes it as a batched per-leaf kernel — it introduces **no new
operator tables** (it is a body-mediated evaluation, like L2B with
irregular harmonics).

**Lamb–Helmholtz.** Per `008e`/`008h`, the evaluated velocity coefficient
at degree $n$ reads $\phi_{n+1}$ and $\chi_n$ (no degree shift on $\chi$;
same-degree $S_n^{m\pm1}$ combinations with weights $(n{\mp}m)$, and the
$z$ component $-i\,m\,\chi_n^m S_n^m$). The multipole $\chi$ channel is
carried at $P_\chi=P_\phi+1$ ($P_{\rm active}$ sizing per `008h`), so M2T
consumes exactly the channel degrees the V path already transports;
truncation behavior is consistent with the constant-$P$ model by
construction.

**Normalization and scaling.** M2T inherits the production potential
normalization (unit production source evaluates to $-1/(4\pi r)$; analytic
comparisons multiply by $-4\pi$). No level scaling is involved: the
irregular harmonics are evaluated at the physical displacement, so the
`025` scaling law is not needed (and no per-level table exists to scale).

**Verification oracle.** M2T is exactly the composition
(M2L to a zero-radius target expansion at $x_t$) $\to$ (evaluate at its
own center): $u = [K(x_t-c_B)\,M]_0^0$ evaluated trivially. This exact
identity gives rows `040`/`041` a parity oracle against the validated
dense M2L operators for every channel, including $\chi$.

### 4.2 S2L (source-to-local)

**Definition (scalar point source).** For an X pair (source leaf $B$,
target cell $A$ with local expansion about $c_A$), each source body at
$x_s$ with production strength $\hat q$ accumulates

$$
L_n^m \;\mathrel{+}=\; -(-1)^{n+m}\, \hat q\, \overline{S_n^m(x_s-c_A)},
\qquad 0\le m\le n\le P,
$$

the mirror of the production P2M rule
$M_n^m = -(-1)^{n+m}\,\hat q\,\overline{R_n^m(x_s-c_s)}$ with the
irregular harmonic in place of the regular one. The resulting local
expansion is evaluated by the unchanged L2L/L2B machinery. The scalar
identity (both the P2M/evaluate pair and this S2L/evaluate-local pair
converge to the analytic $-1/r$ potential; the two agree term-by-term via
the addition theorem) is verified numerically at $P=4$ and $P=8$ by the
validation script with self-contained harmonics.

**Lamb–Helmholtz (vector/vortex sources).** The channel content of a
source body is defined by the same strength-to-channel map production B2M
uses (`src/bodytomultipole.jl`, `mirrored_source_to_vortex!` for
`Point{Vortex}`), applied to irregular harmonics of the displacement
$x_s-c_A$ in place of regular harmonics, with $\chi$ rows carried through
$P_\phi+1$ including the `008h` neighbor row. Cross-channel structure is
unchanged from B2M because both expansions represent the same physical
field decomposition ($\phi$ scalar channel, $\chi$ Lamb–Helmholtz gauge
channel about the *local* center's $z$-axis convention, which S2L shares
with M2L's output — both are locals about $c_A$).

**Verification oracle.** S2L is exactly the composition
(P2M about a zero-radius source cell at $x_s$) $\to$ (M2L over the
displacement $x_s\to c_A$): $L = K(c_A - x_s)\,M_{\rm point}$, where the
point multipole is *finite* (degree 0 for scalar sources, degree $\le1$
for point vortices), so the composition has no source-truncation error and
is an exact per-channel oracle through the validated dense M2L operators.
Rows `040`/`041` must include this parity test (both channels, both
precisions, $P=4$ and $P=8$).

**Cost.** M2T and S2L are $O((P{+}1)^2)$ per (body, cell) entry — the same
complexity class as L2B/B2M — versus $O((P{+}1)^3)$ for a V route and
$O(K)$ per body for a direct partner cell. The cost model (§6) prices them
with the L2B/B2M constants.

### 4.3 Error bounds consistent with the constant-$P$ model

Let a W pair (leaf $A$, source cell $B$ at level $\ell_B$) be emitted. By
the §2.1 far test, every target body $x_t\in A$ lies in a level-$\ell_B$
tile $o(x_t)\notin N_{q_{\ell_B}}$ relative to $B$. The M2T truncation
error at $x_t$ is the multipole tail of a source cell of half-width
$w_{\ell_B}$ evaluated at a point of a far cell at offset $o(x_t)$ — the
*source half* of the geometry the `008d` bound conservatively covers with
both halves:

$$
\bigl|u(x_t) - u_P(x_t)\bigr| \;\le\; B\!\left(P,\,o(x_t),\,A_B\right)
\;\le\; B\!\left(P,\,o_{\min},\,A_B\right),
$$

where $A_B$ is the source budget of $B$ and $o_{\min}$ the
minimum-norm separated offset ($\|o\|^2=q+1$ shell representative;
$B$ is monotone decreasing in $\|o\|$ for $c>2$ — verified numerically in
the evidence tables at $P=4$ and $P=8$). This is *the same worst-case
bound as a V interaction at level $\ell_B$*: W entries respect the same
accuracy target as V, with margin (no target-side truncation occurs at
all). Dually, an X pair's S2L has no *source* truncation (the sources are
evaluated exactly) and only the local tail at the target cell's level
$\ell_A$, bounded by the target half of the same $B(P,o,A)$ at
$\ell_A$-scale geometry. For Lamb–Helmholtz, apply the `008h` channel
tails: $B_\phi = B(P_\phi,o,A_\phi)$, $B_\chi = B(P_\phi{+}1,o,A_\chi)$,
combined as $B_{\rm LH}=B_\phi+(1+2R)B_\chi$; the $\chi$ order rule makes
the W/X channel budgets identical to the V-path budgets.

Two conservative statements complete the model:

1. Because W/X error is bounded by the V bound at the same level and
   minimal offset, any $\varepsilon$ schedule (per-level radii) that makes
   the V path admissible under `008d` makes W/X admissible with margin.
2. The per-pair count is also conservative: each body pair is covered
   exactly once (§3), so error contributions add exactly as in the uniform
   path's budget — no pair is double-counted.

### 4.4 The $c\le2$ exclusion

The `008d` bound requires $c=2\|o\|/\sqrt3>2$, i.e. $\|o\|^2>3$. For
$q=3$ the minimal separated shell is $\|o\|^2=4$ ($c\approx2.31$), and for
$q=12$ it is $\|o\|^2=13$: every admissible V/W/X offset at both radii
satisfies $c>2$, so the bound is finite everywhere it is applied. (The
$\|o\|^2=3$ shell — infinite bound — is inside both near sets, exactly as
in `025`.)

## 5. Per-cell $\sigma$ geometry gate

### 5.1 The contract and the global gate it replaces

The `031a` §5.1 contract: every ordered pair with $r\le\rho_t\sigma_s$
($\sigma_s$ the *source* smoothing radius) must be evaluated by the
regularized direct kernel; pairs beyond the cutoff may use the singular
kernel (the omitted Gaussian tail is inside the $\varepsilon=10^{-3}$
budget by the `031a` tail bound and half-budget rule). The shipped uniform
path enforces this globally
(`_direct_kernel_geometry_gate!`): with $g_{\min}$ the minimum AABB gap
(in leaf-cell units) over separated offsets
($g_{\min}=1$ for $q=3$, $\sqrt5$ for $q=12$), it *throws* unless

$$
g_{\min}\, h_{\rm leaf} \;>\; \rho_t\,\sigma_{\max},
$$

equivalently $2^\ell < g_{\min}L_{\rm box}/(\rho_t\sigma_{\max})$: one
locally large $\sigma$ forces a *globally* shallow tree or an outright
failure. This section replaces that global form.

### 5.2 Per-node $\sigma_{\max}$ and the gated far predicate

Define, by one upward sweep (the same reduction pattern as M2M),

$$
\sigma_{\max}(B) \;=\; \max_{s\in\mathrm{sub}(B)} \sigma_s
$$

for every node $B$. Augment the DTR far predicate: a geometrically
separated pair $(A,B)$ is **admissible** iff

$$
\mathrm{gap}(A,B) \;\ge\; \rho_t\,\sigma_{\max}(B),
$$

where $\mathrm{gap}(A,B)$ is the physical AABB gap between the two closed
cell boxes (for same-level cells this is the `025` lattice gap
$\mathrm{gap}(o)\,\Delta_\ell$ with
$\mathrm{gap}(o)=\sqrt{\textstyle\sum_i \max(|o_i|-1,0)^2}$; for mixed
levels it is the per-axis clamp form). A separated pair that fails the
test is **demoted, stickily**: the DTR continues descent following the
ordinary near-pair split rules, and the *entire* descendant pair set of a
demoted pair terminates in U — no V/W/X emission occurs below a demotion
(the lineage carries one bit). Only the source side is tested because the
kernel's regularization variable is $\rho=r/\sigma_{\rm src}$ (the
production pair kernels read only the source $\sigma$).

Stickiness is what preserves §2.4: Invariant 2 (every emitted V pair has
*geometrically* near parents) survives the gate because V is only ever
emitted on never-demoted recursion paths, where "near" coincides with the
geometric predicate. A non-sticky variant — re-admitting descendants of a
demoted pair to V once their refined $\sigma_{\max}$ passes the gate —
emits V pairs whose parents are geometrically separated, i.e. offsets
*outside* the `025` phase-table class set (measured before this
correction: up to Chebyshev reach 11 on the one-fat-core test fields),
violating the no-new-tables constraint. See §5.4 for the recorded
alternative that recovers that lost far-field work without new tables.

Precompute $\rho_t\sigma_{\max}(B)$ per node; the gate is then one
comparison per classified pair in the §2.7 sweep.

### 5.3 Correctness theorem

**Theorem.** With the demotion gate active, (i) exact-once coverage holds
unchanged, and (ii) every ordered body pair with $r\le\rho_t\sigma_s$ is
covered by U.

*Proof.* (i) is immediate from §3.2: the proof is independent of the
near/far predicate's form, and sticky demotion only changes which pairs
descend (the lineage bit alters classification, never the partition
structure of an expansion step). (ii): suppose the pair $(t,s)$ is
covered by an expansion bucket V/W/X $(A,B)$ with $s\in\mathrm{sub}(B)$,
$t\in\mathrm{sub}(A)$. Expansion buckets are emitted only on non-demoted
paths, so the pair itself passed the gate:
$r \ge \mathrm{gap}(A,B) \ge \rho_t\,\sigma_{\max}(B) \ge \rho_t\sigma_s$
(the first inequality because $t$ and $s$ lie inside the respective closed
boxes). Contrapositive: a cutoff pair cannot be covered by an expansion
bucket, and by (i) it is covered somewhere — hence in U. $\square$

Combined with the `031a` tail bound (singular-vs-regularized discrepancy
beyond $\rho_t$ inside the half budget) and §4.3 (truncation inside the
stencil budget), the phase accuracy gate (velocity RMS $\le10^{-3}$) is
implied under the same budget split as the uniform path — with the global
throw eliminated: *any* $\sigma$ field is admissible, in the worst case by
demoting everything near a fat-$\sigma$ region to direct (which is the
regularized-everywhere limit, correct by `032`'s contract).

**Cost locality.** Sticky demotion converts the whole demoted pair set
$\mathrm{sub}(A)\times\mathrm{sub}(B)$ to direct work, including sources
whose own $\sigma$ is small. The over-cost is nevertheless *local*: a
demotion at $(A,B)$ requires $\mathrm{gap}(A,B) < \rho_t\sigma_{\max}(B)$,
so demoted work is confined to the physical
$\rho_t\sigma_{\max}$-neighborhood of the subtree that actually holds the
fat sources — regions the `031a` contract forces (mostly) direct anyway —
and the split veto below keeps such subtrees coarse, bounding the demoted
volume. Away from fat-$\sigma$ subtrees the gate never fires and nothing
changes. The gate is therefore still per-cell in the sense that matters:
one fat core costs direct work only in its own neighborhood, not a
globally shallow tree. The gated-vs-ungated direct-pair counts in
`data/adaptive_octree/sigma_gate_contract.csv` quantify the over-cost on
the one-fat-core and two-decade-heterogeneous test fields.

### 5.4 Split veto, hysteresis, and the E2 mechanism

Demotion alone can waste tree depth: leaves refined far below the locally
admissible scale generate large demoted-pair frontiers before terminating
in U. The complementary *split veto* applies the same arithmetic in
construction (§1.2 step 2): do not split a cell at level $\ell$ if

$$
g_{\min}\,\Delta_{\ell+1} \;<\; \rho_t\,\sigma_{\max}(\text{cell}),
$$

i.e. if the children's own separated shell could not clear the cutoff for
the cell's *own* sources. This is the local analog of the global depth
bound ($\Delta_\ell \ge \rho_t\sigma_{\max}({\rm cell})/g_{\min}$ locally
instead of $\sigma_{\max}$ globally). The veto is a performance heuristic
— correctness never depends on it (the demotion gate is the backstop), and
it must not veto *balance* splits (balance keeps priority; demotion covers
any resulting inadmissible geometry). Recommended default: veto ON for
population splits, OFF for balance splits, with the demotion gate always
armed.

**Implementation note (039 measurement, pending user ratification).** The
veto as displayed keys on the cell's *own subtree* $\sigma_{\max}$, so a
single fat-$\sigma$ body vetoes every one of its ancestors' population
splits up to the root: on the one-fat-core field ($n=1500$,
$\sigma=3\times10^{-4}$ background plus one $0.15$, $\rho_t=4.789$) the
$q=3$ tree ($g_{\min}=1$) collapses to a single root leaf — globally
direct, the very pathology §5 exists to remove — while $q=12$
($g_{\min}=\sqrt5$) only coarsens locally (411 vs 424 leaves). Sticky
demotion alone reproduces the cost-locality claim (116/183 demotions at
$q=3/12$, exact-once and §5.3 contract intact). Row `039` therefore ships
the veto **default OFF** (`AdaptiveTreePolicy(split_veto=...)`), retained
as an option for spatially smooth $\sigma$ fields; a locality-limited
veto (population- or quantile-bounded) is a candidate re-derivation if
measurement in `040` justifies it.

**Recorded alternative (user decision required to adopt).** The far-field
work sticky demotion forgoes could be recovered *without* new operator
tables by re-admitting demoted-descendant far pairs only as body-mediated
M2T/S2L entries (the §4.3 bound already covers their geometry, and those
operators are table-free): sources in the small-$\sigma$ octants of a
demoted subtree would then still be accelerated. The price is M2T/S2L
with internal-node partners (per-body cost $\propto|\mathrm{sub}|$), a
larger W/X capacity term, and a more intricate lineage rule. Sticky
demotion was chosen for this derivation as the theoretically cleanest
form (unconditional table claim, one-bit lineage, simplest proofs); the
re-admission variant is recorded here and in the campaign decision log as
an optimization for user ratification, implementable inside `039`–`040`
without changing this row's theorems.

Because the veto/demotion pair keys off the *live* per-node
$\sigma_{\max}$, `CoreSpreading`-grown $\sigma$ degrades geometry
gracefully and locally (leaves coarsen near grown cores at the next
rebuild; demotion covers drift between rebuilds, §7). This construction
appears to naturally subsume the held E2 mechanism (the per-cell
admissibility replacing the global adequacy test); per the standing
instruction this is noted in the campaign decision log and the E2
disposition item itself remains open for the user.

## 6. Cost and capacity model

### 6.1 Work terms

With $P$ fixed, per-unit costs (calibration constants, measured on the
target hardware in `039`–`041`): $c_d$ per direct body pair, $c_v$ per V
route (one $O((P{+}1)^3)$ class-batched M2L application), $c_{wx}$ per
M2T/S2L body evaluation ($O((P{+}1)^2)$, the L2B/B2M constant), $c_b$ per
body for B2M/L2B, $c_n$ per node for M2M/L2L. Total work:

$$
T \;=\; c_d\!\!\sum_{(A,B)\in U}\!\! |A||B|
\;+\; c_v |V|
\;+\; c_{wx}\!\left(\sum_{(A,B)\in W}\!\!|A| + \sum_{(A,B)\in X}\!\!|B|\right)
\;+\; c_b n \;+\; c_n N_{\rm node}.
$$

### 6.2 List-size bounds

Let $N_{\rm leaf}$, $N_{\rm node}$ be leaf/node counts and let leaves obey
$|A|\le K_{\max}$ (population splits; depth-capped and balance leaves obey
it a fortiori or are handled by the $\ell_{\max}$ cap term).

- **U.** A leaf's same-level near partners number $\le|N_q|$. Coarser
  near partners: a coarser cell is near $A$ only if one of its tiles is in
  $A$'s near shell, and disjoint coarser cells give $\le|N_q|$ per coarser
  level, $\le\ell_{\max}|N_q|$ total (with balance and $q=3$: one coarser
  level, $\le|N_3|$). Finer near partners at $k$ levels down lie in the
  $\sqrt q$-shell of $A$'s tile block: at most
  $(2^k+2\lceil\sqrt q\rceil)^3-\max(2^k-2\lceil\sqrt q\rceil,0)^3 =
  O(\lceil\sqrt q\rceil\,4^k)$ cells. Under balance with $q=3$ the finer
  window is $k=1$ and the count collapses to $\le|N_3|$-scale; for $q=12$
  the capacity formula keeps the window sum explicitly. Direct work per
  leaf is bounded by $K_{\max}$ times the partner-body sum; at balanced
  occupancy $b$, the expected direct term is $\approx |N_q|\,n\,b$ —
  identical in form to the uniform model (`025` §Cost), with $b$ now
  *controlled by construction* ($b\le K_{\max}$) instead of emergent. This
  is the design's central claim: the fat-cell term
  $\max_A|A|^2$ that serialized the rotor nearfield is replaced by
  $K_{\max}\cdot$(partner bodies), a tunable constant.
- **V.** Every V pair has near parents, so charging pairs to the source
  parent gives $|V|\le 8|V_{\rm push}|$-scale per internal node:
  $|V| \le |V^{q}_{\rm push,max}|\cdot N_{\rm node}$ with the per-phase
  cardinalities 189 ($q=3$) / 1253 ($q=12$); the standard full-octree
  estimate $|V|\lesssim\frac87|V_{\rm push}|N_{\rm leaf}$ applies at
  uniform occupancy.
- **W/X.** Charging a W entry to its (near) parent pair: each near pair of
  a leaf spawns $\le8$ W candidates, so
  $|W|\le 8\,|U^{\rm cand}|$-scale; duality gives $|X|=|W|$ under
  role exchange. Their *work* terms are per-body and bounded by
  $K_{\max}$ per entry.

### 6.3 Node capacity bounds

Population splits: at any one level the split nodes are disjoint and each
holds $>K_{\max}$ bodies, so there are at most
$\lceil n/(K_{\max}{+}1)\rceil$ per level and

$$
N_{\rm split} \;\le\; \ell_{\max}\left\lceil\frac{n}{K_{\max}+1}\right\rceil,
\qquad
N_{\rm node} \;\le\; 1 + 8N_{\rm split}.
$$

Balance splits add nodes not covered by the population argument; the
literature constant-factor bound for 2:1 balance closure motivates a
capacity *allowance* factor $\beta_{\rm bal}$ (default 2) rather than a
tight formula, with the measured inflation recorded per case in the
evidence tables (observed well below 2 on all five test distributions).

### 6.4 Capacity formulas for the `RadixFMMCache` contract

For `max_n_bodies` $=n_{\max}$, split threshold $K_{\max}$, depth cap
$\ell_{\max}$, near radius $q$, balance allowance $\beta_{\rm bal}$:

$$
\begin{aligned}
\mathrm{cap_{node}} &= \beta_{\rm bal}\left(1 +
   8\,\ell_{\max}\left\lceil\tfrac{n_{\max}}{K_{\max}+1}\right\rceil\right),
&\mathrm{cap_{leaf}} &= \mathrm{cap_{node}},\\
\mathrm{cap_{V}} &= |V^{q}_{\rm push,max}|\cdot \mathrm{cap_{node}},
&\mathrm{cap_{U}} &= \Bigl(\textstyle\sum_{k}\!c_U(q,k)\Bigr)\,\mathrm{cap_{leaf}},\\
\mathrm{cap_{W}} = \mathrm{cap_{X}} &= 8\,\mathrm{cap_{U}},
&\mathrm{cap_{frontier}} &= \max_{\rm round}|{\rm frontier}| \le \mathrm{cap_{V}} ,
\end{aligned}
$$

with $c_U(q,k)$ the per-level-window partner-count bound of §6.2
(collapsing to $\approx 3|N_q|$ under balance at $q=3$). These are
construction-time capacities: allocate once from $n_{\max}$, assert on
overflow (the existing `RadixFMMCache` invariant style: capacity violation
is an error, never a silent realloc), zero per-step allocation. The
formulas are deliberately conservative; the evidence tables record
measured-to-capacity ratios so `039` can tighten constants with data
rather than re-derivation.

Expansion storage is $\mathrm{cap_{node}}\times$ the `007` per-node size —
note this *grows* relative to the uniform path's occupied-cell count at
equal leaf scale (ancestors at every level), but the adaptive leaf count
itself is smaller wherever density is nonuniform; the evidence tables
report both.

### 6.5 Expected behavior and the multi-scale case

The optimum $K_{\max}$ balances $c_d|N_q|\,b$ per body against the
per-cell far cost; the qualitative law matches `025`
($\sqrt{|N||V|}$-scale), but adaptivity holds $b$ near $K_{\max}$
*everywhere* rather than only on uniform fields. On a field with local
density contrast $C_\rho$, the uniform grid at its best single depth pays
either $\propto C_\rho$ extra direct work (fat cells) or $\propto C_\rho$
extra cells (fine everywhere); the adaptive tree pays neither — its leaf
scale tracks $n_{\rm local}^{1/3}$.

The validation script generates counted evidence (no timing):
per-case tables of $N_{\rm leaf}$, $N_{\rm node}$, balance inflation,
$\max_A|A|$, U body pairs, V routes, W/X entries/evaluations vs
$K_{\max}\in\{8,16,32,64,128\}$, against the uniform grid at every depth
$\ell\in\{2..6\}$ — on the uniform cube, the embedded-cluster multi-scale
cases (30× and 100× contrast, per the task file), the rotor-like filament,
and the adversarial two-cluster case
(`data/adaptive_octree/cost_model_counts.csv`, $n=3000$ per case,
model constants $c_d{=}1$, $c_v{=}600$, $c_{wx}{=}30$). Two counted
findings anchor the model:

1. **Multi-scale win at equal worst-cell population.** The GPU failure
   mechanism is the fat cell ($\max_A|A|$, warp-serialized $O(K^2)$).
   Comparing configurations with *equal* $\max_A|A|$: on
   `multiscale100`, $q{=}3$, the uniform grid needs $\ell{=}4$ to reach
   $\max_A|A|=56$ at modeled cost $8.7\times10^7$, while the adaptive tree
   reaches the same $\max_A|A|$ at $K_{\max}{=}64$ for $1.1\times10^7$ —
   **7.8× less modeled work**; at $q{=}12$ the same comparison gives
   **20.7×** ($3.6\times10^8$ vs $1.7\times10^7$). On the filament,
   $q{=}3$ at $\max_A|A|=64$: **5.1×** ($2.9\times10^7$ vs
   $5.8\times10^6$). This is the counted analog of the measured 7.16x
   rotor finding, and the contrast grows with $n$ at fixed contrast ratio.
2. **Uniform non-regression is exact.** On the uniform cube the adaptive
   tree collapses to a single depth and its rows *equal* the uniform-grid
   rows entry for entry ($K_{\max}{=}64\leftrightarrow\ell{=}2$,
   $K_{\max}{=}16,32\leftrightarrow\ell{=}3$; W/X empty, V identical),
   the counted form of §3.3.

## 7. Refresh and rebuild policy

The tree is a function of body positions (and $\sigma$); motion
invalidates it gradually. Three tiers:

1. **Per-step refresh (leaf set frozen).** Recompute full-depth keys and
   re-sort (the existing device sort); recompute each node's body range by
   binary search of its key interval (prefix ranges over the sorted keys);
   recompute per-node $\sigma_{\max}$ (one upward sweep). Everything
   structural — node table, U/V/W/X lists, $(\ell,o)$ classes, operator
   tables, occupancy-epoch caches, capacities — is reused unchanged.
   Validity conditions, checked by cheap reductions each step:
   - every body maps into some current leaf's key interval. Because the
     occupied leaf set does not cover previously-empty space, a body can
     drift into a region no leaf covers; such an unmatched key breaks the
     coverage premise of §3 and is an immediate rebuild trigger (detected
     for free during the range binary searches: total matched range
     lengths must sum to $n$);
   - leaf populations within the hysteresis bound $|A|\le K_{\rm hi}$
     (default $K_{\rm hi}=2K_{\max}$) — prevents slow fat-cell regrowth;
   - the per-node gate margin holds: $\rho_t\,\sigma_{\max}(B) \le
     m(B)$, where $m(B)=\min_{\text{emitted buckets on }B}\mathrm{gap}$
     is stored at list build (one comparison per node; covers
     `CoreSpreading` growth between rebuilds).
2. **Rebuild (structure refresh).** Full §1 construction + §2 lists at
   unchanged capacities. Triggered by any validity failure, or every
   $T_{\rm rebuild}$ steps as a policy floor. Cost is sort/scan/compact —
   the same primitive budget as the existing refresh path, amortized over
   the validity window; the expected window is
   $\sim\Delta_{\rm leaf,min}/(v_{\max}\,\mathrm{d}t)$ steps.
3. **Recenter (`recenter!`).** Changing the root center/half-width changes
   every key: always a full rebuild. Policy: recenter only on rebuild
   epochs (drift-triggered, hysteresis on the root-box fit), so
   `recenter!` never invalidates a frozen leaf set mid-window. This
   matches the existing occupancy-epoch semantics: bump the epoch counter
   on every rebuild; consumers key caches by epoch exactly as today.

Capacity note: rebuilds never reallocate — all §6.4 capacities are
functions of $n_{\max}$, $K_{\max}$, $\ell_{\max}$, $q$ only. A rebuild
that would exceed capacity (possible only if the conservative formulas
were manually overridden) is an error by the `RadixFMMCache` contract.

## 8. Verification artifacts and reproducibility

```sh
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/adaptive_octree_verify.jl
```

writes to `MATRIX_OPERATOR_REFACTOR/data/adaptive_octree/`:

- `exact_once_coverage.csv` — §3 exact-once + balance + V-class + W/X
  level statistics per (case, $q$, $K_{\max}$, balanced);
- `sigma_gate_contract.csv` — §5 sticky-demotion counts, cutoff-coverage
  contract checks, and gated-list V-class/phase-table membership + W/X
  structure (uniform-small, heterogeneous, one-fat-core $\sigma$);
- `cost_model_counts.csv` — §6 counted work terms, adaptive vs uniform
  depths;
- `constant_p_bound_consistency.csv` — §4.3/§4.4 bound values and
  monotonicity at $P=4$ and $P=8$;
- `s2l_m2t_convergence.csv` — §4.1/§4.2 scalar M2T and S2L convergence to
  the analytic potential at $P=4$ and $P=8$ (self-contained Gumerov-form
  harmonics);
- `summary.txt` — pass/fail roll-up.

The script is stdlib-only, single-threaded, deterministic (fixed seeds),
and does not import or mutate production FastMultipole code.

## 9. Consequences for rows 039–041 (implementation contract)

- `039` (host construction): §1 construction + §1.4 balance + §2.7 lists,
  V lists in the existing $(\ell,o)$ class format; §6.4 capacities;
  uniform-limit parity (§3.3) and brute-force exact-once as tests
  (clustered + uniform + P=4-sized cases).
- `040` (host lifecycle): §2.6 pipeline; M2T/S2L kernels per §4 with the
  M2L-composition oracles as parity tests ($\phi$+$\chi$, both
  precisions, $P=4$ and $P=8$); §5 gate with *sticky* demotion default-on
  (the §5.4 re-admission alternative only after user ratification);
  split-veto default **OFF** per the §5.4 implementation note (the `039`
  measured deviation, pending user ratification — corrected here on the
  `040` touch; an earlier draft of this line said default-on); accuracy
  gates per the phase contract.
- `041` (CUDA): §1.2/§2.7 as flag/scan/compact kernels; sorted-key binary
  search replaces the dense $\Sigma 8^L$ occupancy table (record whether
  this lifts the uniform path's $\ell\le8$ cap); §7 epochs.
- No production default changes are proposed by this row; the adaptive
  path is opt-in until `041a` measurement and user approval.
