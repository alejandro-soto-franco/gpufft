#!/bin/bash
# Rust -> Charon -> Aeneas -> Lean for gpufft/src/vulkan/stride_layout.rs. Fails on a test
# failure, on drift between the Rust and the committed Lean translation, and on any axiom
# beyond Lean's standard three. Tools: the Aeneas nightly-2026.09.24-557f7a1 release at
# $AENEAS_HOME (default ~/local/aeneas), which bundles Charon at the commit Aeneas pins.
set -euo pipefail
cd "$(dirname "$0")"
A=${AENEAS_HOME:-$HOME/local/aeneas}

echo "== Rust tests"
(cd .. && cargo test --quiet -p gpufft --lib vulkan::stride_layout)

echo "== Charon"
(cd stride_core && "$A/charon" cargo --preset=aeneas --dest-file ../stride_core.llbc)

echo "== Aeneas"
tmp=$(mktemp -d)
"$A/aeneas" -backend lean stride_core.llbc -dest "$tmp" -split-files -loops-to-rec >/dev/null
rm -f stride_core.llbc
for f in Funs.lean Types.lean; do
  diff -q "$tmp/$f" "lean/StrideCore/$f" >/dev/null || {
    rm -rf "$tmp"
    echo "generated $f differs from lean/StrideCore/$f: regenerate and re-prove"; exit 1; }
done
rm -rf "$tmp"

echo "== Lean"
(cd lean && lake build)
axioms=$(cd lean && lake env lean CheckAxioms.lean)
echo "$axioms"
if echo "$axioms" | grep -v "depends on axioms: \[propext, Classical.choice, Quot.sound\]$" | grep -q .; then
  echo "unexpected axiom or missing theorem"; exit 1
fi
echo "verified"
