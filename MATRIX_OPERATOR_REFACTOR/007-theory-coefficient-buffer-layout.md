# 007 Theory Coefficient Buffer Layout

## Objective

Specify coefficient-buffer layout, indexing, and typed view requirements for
the explicit operator layer.

## Dependencies

- `005-theory-full-m2l-composition.md`
- `006-theory-m2m-l2l-extensions.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Current coefficient storage, indexing, and scratch-buffer code

## Artifacts or Production Surface

- `theory/coefficient-buffer-layout.md`
- `scripts/coefficient_buffer_layout_*.jl`
- `data/coefficient_buffer_layout/`

## Deliverables

- Proposed flat-buffer layout and typed views
- Mapping to current logical coefficient layout
- Scratch, cache, and aliasing requirements

## Verification

Validate indexing maps with deterministic coefficient round trips and layout
examples. Record commands, tolerances where relevant, and result summaries.

## Approval Notes

To be filled by a different agent after artifacts and verification are
complete.
