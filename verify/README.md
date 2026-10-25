# Verified stride-copy dispatch

Vulkan R2C and C2R plans pad and strip the innermost axis with
`shaders/stride_copy.comp`, which indexes its two storage buffers in `u32`
arithmetic. `gpufft/src/vulkan/stride_layout.rs` computes the shader's push
constants, the word counts of its bindings and its workgroup count, and accepts a
dispatch only when both bindings fit the device's `maxStorageBufferRange` and the
dispatch fits its `maxComputeWorkGroupCount[0]`. Charon and Aeneas translate that
file to Lean 4, and `lean/StrideProofs.lean` proves the following for every input:

- `plan_spec`: `plan` never panics or overflows. A dispatch it accepts has bindings of
  `n_rows * stride` words within the byte range and enough workgroups for every word,
  and needs no more workgroups than the limit allows.
- `invocation_spec`: `invocation` mirrors the shader's `main`. On an accepted dispatch
  its `u32` arithmetic never wraps for any invocation id. Every invocation that does
  not return early copies word `tid % row_uints` of row `tid / row_uints`, and both
  indices lie inside their bindings.
- `invocation_covers`: word `c` of row `r` is copied by an invocation inside the
  launched workgroups.
- `invocation_writes_disjoint`: no two invocations write the same destination word.
- `real_transform_spec`: `real_transform` accepts a pad and a strip dispatch only
  together. The strip swaps the tight and padded strides of the pad, and the padded
  binding has the same size in both.

Each theorem depends on Lean's three standard axioms alone. `stride_core/` compiles the
shipped `stride_layout.rs` through a `#[path]` include. The proofs therefore cover the
code the plans run. Seven planted bugs each break a proof: five in `plan` and `real_transform`
and two in the shader mirror.

The proofs take `invocation` as a faithful copy of `stride_copy.comp`. Neither tool
reads GLSL, and a change to one needs the same change to the other.

## Running

`./run.sh` runs the Rust tests, extracts with Charon, translates with Aeneas, compares
the result with the committed Lean in `lean/StrideCore/`, builds the proofs and checks
their axioms. It fails on a test failure, on drift between the Rust and the committed
translation and on any axiom beyond the standard three.

Tools: the Aeneas nightly `nightly-2026.09.24-557f7a1` (override its location with
`AENEAS_HOME`), which bundles the Charon its Lean backend expects. Charon needs the
Rust toolchain `nightly-2026-09-17`. The Lean project uses Lean and Mathlib v4.31.0 and
takes Aeneas from git at the same commit.
