#!/usr/bin/env bash
#
# Task 024a -- regenerate every benchmark figure from the committed CSVs.
#
#   bash MATRIX_OPERATOR_REFACTOR/scripts/figures_024a_build.sh
#
# Stage 1: figures_024a_prepare.jl reads data/<campaign>/*.csv and writes tidy
#          per-panel tables into data/figures/tables/.
# Stage 2: latexmk compiles each data/figures/fig*.tex to PDF, run from that
#          directory so the relative tables/*.csv paths resolve.
# Stage 3: optional PNG rasterization when a converter is available.
#
# Requirements: julia (stdlib only -- no package environment is touched) and a
# TeX distribution with pgfplots, pgfplotstable and latexmk.  Verified against
# Julia 1.12.5 and TeX Live 2025 (pgfplots 1.18-compatible).
#
# The task 024b campaign completed on 2026-07-28, so fig09 now builds strictly
# by default -- no environment variable is needed.  FM024B_ALLOW_PARTIAL=true
# remains only as a resume aid for an incomplete re-run: it relaxes the 024b
# completeness gates, plots only the particle counts with all four comparison
# modes, and labels the figure PROVISIONAL.  Figures 1-8 (task 024a) are
# unaffected either way.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFACTOR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIG_DIR="${REFACTOR_DIR}/data/figures"

command -v julia   >/dev/null || { echo "024a: julia not found" >&2; exit 1; }
command -v latexmk >/dev/null || { echo "024a: latexmk not found" >&2; exit 1; }
kpsewhich pgfplots.sty >/dev/null || { echo "024a: pgfplots.sty not found" >&2; exit 1; }

echo "== 024a stage 1: preparing figure tables =="
julia "${SCRIPT_DIR}/figures_024a_prepare.jl"

echo "== 024a stage 2: compiling figures =="
cd "${FIG_DIR}"
shopt -s nullglob
sources=(fig*.tex)
(( ${#sources[@]} > 0 )) || { echo "024a: no fig*.tex sources in ${FIG_DIR}" >&2; exit 1; }
for tex in "${sources[@]}"; do
    echo "-- ${tex}"
    latexmk -pdf -interaction=nonstopmode -halt-on-error "${tex}" >/dev/null
done
latexmk -c >/dev/null   # drop .aux/.log/.fls build residue, keep the PDFs

echo "== 024a stage 3: PNG copies =="
if command -v magick >/dev/null; then
    for pdf in fig*.pdf; do
        magick -density 200 "${pdf}" -background white -alpha remove "${pdf%.pdf}.png"
    done
elif command -v pdftoppm >/dev/null; then
    for pdf in fig*.pdf; do
        pdftoppm -png -r 200 -singlefile "${pdf}" "${pdf%.pdf}"
    done
else
    echo "   no rasterizer (magick / pdftoppm) found -- shipping PDF only"
fi

echo
echo "024a: figures in ${FIG_DIR}"
ls -1 fig*.pdf
