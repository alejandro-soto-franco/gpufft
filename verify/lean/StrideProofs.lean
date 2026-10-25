import Aeneas
import StrideCore
open Aeneas Aeneas.Std Result

namespace StrideCore

/-- An accepted dispatch: nonempty rows no longer than either stride, bindings of
`n_rows * stride` words that fit in a `u32` byte range, and enough workgroups for
every word of every row. -/
def Valid (p : stride_core.stride_layout.StrideCopy) : Prop :=
  0 < p.row_uints.val ∧ 0 < p.n_rows.val ∧
  p.row_uints.val ≤ p.src_stride_uints.val ∧ p.row_uints.val ≤ p.dst_stride_uints.val ∧
  p.src_words.val = p.n_rows.val * p.src_stride_uints.val ∧
  p.dst_words.val = p.n_rows.val * p.dst_stride_uints.val ∧
  p.src_words.val * 4 ≤ U32.max ∧ p.dst_words.val * 4 ≤ U32.max ∧
  p.row_uints.val * p.n_rows.val ≤ p.groups_x.val * 64

end StrideCore

namespace stride_core
open StrideCore

theorem local_size_u64 : (UScalar.cast .U64 stride_layout.LOCAL_SIZE_X).val = 64 := by
  simp [stride_layout.LOCAL_SIZE_X]

@[step]
theorem stride_layout.plan_spec (row src dst n : U64) (range groups : U32) :
    stride_layout.plan row src dst n range groups ⦃ r =>
      ∀ p, r = some p → Valid p ∧
        p.row_uints.val = row.val ∧ p.src_stride_uints.val = src.val ∧
        p.dst_stride_uints.val = dst.val ∧ p.n_rows.val = n.val ∧
        p.src_words.val * 4 ≤ range.val ∧ p.dst_words.val * 4 ≤ range.val ∧
        p.groups_x.val ≤ groups.val ⦄ := by
  have hR : range.val ≤ U32.max := by scalar_tac
  have hRw : range.val / 4 < 2 ^ 30 := by
    have : U32.max = 2 ^ 32 - 1 := by simp [U32.max, U32.numBits]
    omega
  unfold stride_layout.plan
  step*
  all_goals
    try simp only [max_words_post, U32.cast_U64_val_eq, i_post] at *
  all_goals
    try have k1 : n.val * src.val ≤ range.val / 4 :=
      (Nat.le_div_iff_mul_le (by scalar_tac)).mp (by rw [← i1_post]; scalar_tac)
  all_goals
    try have k2 : n.val * dst.val ≤ range.val / 4 :=
      (Nat.le_div_iff_mul_le (by scalar_tac)).mp (by rw [← i2_post]; scalar_tac)
  all_goals
    try have k3 : row.val * n.val ≤ n.val * src.val :=
      Nat.mul_comm src.val n.val ▸ Nat.mul_le_mul_right n.val (show row.val ≤ src.val by scalar_tac)
  all_goals
    try subst i3_post
  all_goals
    try subst i6_post
  all_goals
    try simp only [local_size_u64] at *
  all_goals
    try have : U64.max = 2 ^ 64 - 1 := (by simp [U64.max, U64.numBits])
  all_goals
    try omega
  all_goals
    try scalar_tac
  -- The accepted branch: every cast to `u32` is in range.
  intro p hp
  clear this
  simp only [Option.some.injEq] at hp
  subst hp
  have c : ∀ x : U64, x.val < 2 ^ 30 → (UScalar.cast .U32 x).val = x.val := fun x hx =>
    UScalar.cast_val_mod_pow_of_inBounds_eq _ _ (by
      have : (2 : Nat) ^ UScalarTy.U32.numBits = 2 ^ 32 := rfl
      omega)
  simp only [gt_iff_lt, not_lt, UScalar.lt_equiv] at *
  simp only [UScalar.eq_equiv] at *
  have hr0 : row.val ≠ 0 := by simpa using ‹¬row.val = (0#u64).val›
  have hn0 : n.val ≠ 0 := by simpa using ‹¬n.val = (0#u64).val›
  have hgx : groups_x.val ≤ groups.val := by rw [U32.cast_U64_val_eq] at i7_post; omega
  have hsrc : src.val ≤ n.val * src.val := Nat.le_mul_of_pos_left _ (by omega)
  have hdst : dst.val ≤ n.val * dst.val := Nat.le_mul_of_pos_left _ (by omega)
  have hn : n.val ≤ n.val * src.val := Nat.le_mul_of_pos_right _ (by omega)
  have hgv : groups_x.val * 64 ≥ total.val := by
    rw [groups_x_post, i5_post, i4_post]; omega
  have e8 : i8.val = row.val := by rw [i8_post]; exact c _ (by omega)
  have e9 : i9.val = src.val := by rw [i9_post]; exact c _ (by omega)
  have e10 : i10.val = dst.val := by rw [i10_post]; exact c _ (by omega)
  have e11 : i11.val = n.val := by rw [i11_post]; exact c _ (by omega)
  have e12 : i12.val = src_words.val := by rw [i12_post]; exact c _ (by omega)
  have e13 : i13.val = dst_words.val := by rw [i13_post]; exact c _ (by omega)
  have e14 : i14.val = groups_x.val := by rw [i14_post]; exact c _ (by omega)
  have hU : U32.max = 2 ^ 32 - 1 := by simp [U32.max, U32.numBits]
  simp only [Valid, e8, e9, e10, e11, e12, e13, e14, true_and]
  refine ⟨⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩,
    by omega, by omega, by omega⟩

/-- The row term of a flat index stays a full row below the binding's end. -/
theorem row_bound (a n m : Nat) (h : a < n) : a * m + m ≤ n * m := by
  have := Nat.mul_le_mul_right m (Nat.succ_le_of_lt h)
  rw [Nat.succ_mul] at this
  exact this

/-- Splitting `t < R * N` into row `t / R` and column `t % R`: the column is what
remains after the row, and with `R ≤ S` the flat index on stride `S` stays below
`N * S`. -/
theorem split_index (t R N S : Nat) (hR : 0 < R) (ht : t < R * N) (hRS : R ≤ S) :
    t / R * R ≤ t ∧ t - t / R * R = t % R ∧ t / R * S + t % R < N * S := by
  have hq : t / R < N := (Nat.div_lt_iff_lt_mul hR).mpr (by rw [Nat.mul_comm]; exact ht)
  have hm : t % R < R := Nat.mod_lt _ hR
  have hb := row_bound _ _ S hq
  have hd := Nat.div_add_mod t R
  rw [Nat.mul_comm] at hd
  refine ⟨Nat.div_mul_le_self t R, by omega, by omega⟩

/-- Every invocation of an accepted dispatch computes its indices in `u32` without
wrapping. An invocation at or past `row_uints * n_rows` returns early. Any other
invocation copies word `tid % row_uints` of row `tid / row_uints`, and both indices
lie inside their bindings. -/
@[step]
theorem stride_layout.invocation_spec (p : stride_layout.StrideCopy) (hp : Valid p)
    (tid : U32) :
    stride_layout.invocation p tid ⦃ r =>
      match r with
      | none => p.row_uints.val * p.n_rows.val ≤ tid.val
      | some (s, d) =>
        s.val = tid.val / p.row_uints.val * p.src_stride_uints.val + tid.val % p.row_uints.val ∧
        d.val = tid.val / p.row_uints.val * p.dst_stride_uints.val + tid.val % p.row_uints.val ∧
        s.val < p.src_words.val ∧ d.val < p.dst_words.val ⦄ := by
  obtain ⟨hr0, hn0, hrs, hrd, hsw, hdw, hsm, hdm, -⟩ := hp
  have hU : U32.max = 2 ^ 32 - 1 := by simp [U32.max, U32.numBits]
  have k3 : p.row_uints.val * p.n_rows.val ≤ p.n_rows.val * p.src_stride_uints.val :=
    Nat.mul_comm p.src_stride_uints.val p.n_rows.val ▸ Nat.mul_le_mul_right _ hrs
  unfold stride_layout.invocation
  step*
  all_goals
    try have ht : tid.val < p.row_uints.val * p.n_rows.val := by scalar_tac
  all_goals
    try have hs := split_index tid.val _ _ _ hr0 ht hrs
  all_goals
    try have hd := split_index tid.val _ _ _ hr0 ht hrd
  all_goals
    try simp only [row_post] at *
  all_goals
    try simp only [i_post] at *
  all_goals
    try simp only [col_post] at *
  all_goals
    try simp only [i1_post] at *
  all_goals
    try simp only [i3_post] at *
  all_goals
    try simp only [i2_post, i4_post]
  all_goals
    generalize tid.val / p.row_uints.val = q at *
  all_goals
    generalize tid.val % p.row_uints.val = m at *
  all_goals omega

/-- Dividing `x * q + r` by `q`, for `r < q`, recovers `x` and `r`. -/
theorem div_mod_of_lt (x q r : Nat) (hr : r < q) : (x * q + r) / q = x ∧ (x * q + r) % q = r := by
  have hq : 0 < q := by omega
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right r x hq, Nat.div_eq_of_lt hr, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hr]

/-- The dispatch covers every word: word `c` of row `r` is invocation
`r * row_uints + c`, which lies inside the launched workgroups and copies source
word `r * src_stride_uints + c` to destination word `r * dst_stride_uints + c`. -/
theorem stride_layout.invocation_covers (p : stride_layout.StrideCopy) (hp : Valid p)
    (r c : Nat) (hr : r < p.n_rows.val) (hc : c < p.row_uints.val) (tid : U32)
    (ht : tid.val = r * p.row_uints.val + c) :
    tid.val < p.groups_x.val * 64 ∧
    stride_layout.invocation p tid ⦃ x => ∃ s d, x = some (s, d) ∧
      s.val = r * p.src_stride_uints.val + c ∧ d.val = r * p.dst_stride_uints.val + c ⦄ := by
  have hb := row_bound _ _ p.row_uints.val hr
  have hg := hp.2.2.2.2.2.2.2.2
  have hdm := div_mod_of_lt r _ c hc
  rw [← ht] at hdm
  refine ⟨by rw [Nat.mul_comm p.row_uints.val] at hg; omega, ?_⟩
  apply WP.spec_mono (stride_layout.invocation_spec p hp tid)
  intro x hx
  match x, hx with
  | none, hx => rw [Nat.mul_comm] at hx; omega
  | some (s, d), ⟨hs, hd, _, _⟩ => exact ⟨s, d, rfl, by rw [hs, hdm.1, hdm.2], by rw [hd, hdm.1, hdm.2]⟩

/-- No two invocations of an accepted dispatch write the same destination word. -/
theorem stride_layout.invocation_writes_disjoint (p : stride_layout.StrideCopy) (hp : Valid p)
    (t1 t2 s1 d1 s2 d2 : U32)
    (h1 : stride_layout.invocation p t1 = ok (some (s1, d1)))
    (h2 : stride_layout.invocation p t2 = ok (some (s2, d2))) (hd : d1 = d2) : t1 = t2 := by
  have hrd := hp.2.2.2.1
  have hr0 := hp.1
  obtain ⟨x1, hx1, hp1⟩ := WP.spec_imp_exists (stride_layout.invocation_spec p hp t1)
  obtain ⟨x2, hx2, hp2⟩ := WP.spec_imp_exists (stride_layout.invocation_spec p hp t2)
  rw [h1, Result.ok.injEq] at hx1
  rw [h2, Result.ok.injEq] at hx2
  subst hx1 hx2
  obtain ⟨-, e1, -, -⟩ := hp1
  obtain ⟨-, e2, -, -⟩ := hp2
  have m1 := div_mod_of_lt (t1.val / p.row_uints.val) _ _
    (Nat.lt_of_lt_of_le (Nat.mod_lt t1.val hr0) hrd)
  have m2 := div_mod_of_lt (t2.val / p.row_uints.val) _ _
    (Nat.lt_of_lt_of_le (Nat.mod_lt t2.val hr0) hrd)
  rw [← e1, hd] at m1
  rw [← e2] at m2
  have hq : t1.val / p.row_uints.val = t2.val / p.row_uints.val := m1.1.symm.trans m2.1
  have hm : t1.val % p.row_uints.val = t2.val % p.row_uints.val := m1.2.symm.trans m2.2
  apply UScalar.eq_of_val_eq
  rw [← Nat.div_add_mod t1.val p.row_uints.val, ← Nat.div_add_mod t2.val p.row_uints.val, hq, hm]

/-- The pad and strip dispatches of a real transform: both accepted, with rows of
`innermost * elem_uints` words on tight and padded strides swapped between them.
The padded binding of one is the padded binding of the other. -/
@[step]
theorem stride_layout.real_transform_spec (innermost : U32) (n : U64) (eu range groups : U32) :
    stride_layout.real_transform innermost n eu range groups ⦃ r =>
      ∀ pad strip, r = some (pad, strip) → Valid pad ∧ Valid strip ∧
        pad.row_uints.val = innermost.val * eu.val ∧
        pad.src_stride_uints.val = innermost.val * eu.val ∧
        pad.dst_stride_uints.val = 2 * (innermost.val / 2 + 1) * eu.val ∧
        strip.row_uints.val = innermost.val * eu.val ∧
        strip.src_stride_uints.val = 2 * (innermost.val / 2 + 1) * eu.val ∧
        strip.dst_stride_uints.val = innermost.val * eu.val ∧
        pad.n_rows.val = n.val ∧ strip.n_rows.val = n.val ∧
        pad.dst_words.val = strip.src_words.val ∧ pad.src_words.val = strip.dst_words.val ∧
        pad.dst_words.val * 4 ≤ range.val ∧ pad.src_words.val * 4 ≤ range.val ∧
        pad.groups_x.val ≤ groups.val ∧ strip.groups_x.val ≤ groups.val ⦄ := by
  have hi : innermost.val ≤ 2 ^ 32 - 1 := by have := innermost.hBounds; simp at this; omega
  have he : eu.val ≤ 2 ^ 32 - 1 := by have := eu.hBounds; simp at this; omega
  have hU : U64.max = 2 ^ 64 - 1 := by simp [U64.max, U64.numBits]
  have k1 : innermost.val * eu.val ≤ (2 ^ 32 - 1) * (2 ^ 32 - 1) := Nat.mul_le_mul hi he
  have k2 : 2 * (innermost.val / 2 + 1) * eu.val ≤ (2 ^ 32 + 1) * (2 ^ 32 - 1) :=
    Nat.mul_le_mul (by omega) he
  unfold stride_layout.real_transform
  step*
  all_goals
    try subst_vars
  all_goals
    try simp only [U32.cast_U64_val_eq] at *
  all_goals
    try have h5 : i5.val = 2 * (innermost.val / 2 + 1) := by omega
  case hmax => rw [h5]; omega
  intro pad' strip' h
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  obtain ⟨vp, pr, ps, pd, pn, psw, pdw, pg⟩ := o_post pad rfl
  obtain ⟨vs, sr, ss, sd, sn, ssw, sdw, sg⟩ := o1_post strip rfl
  have hpad : padded.val = 2 * (innermost.val / 2 + 1) * eu.val := by rw [padded_post, h5]
  have e1 : pad.dst_words.val = n.val * padded.val := by rw [vp.2.2.2.2.2.1, pn, pd]
  have e2 : strip.src_words.val = n.val * padded.val := by rw [vs.2.2.2.2.1, sn, ss]
  have e3 : pad.src_words.val = n.val * tight.val := by rw [vp.2.2.2.2.1, pn, ps]
  have e4 : strip.dst_words.val = n.val * tight.val := by rw [vs.2.2.2.2.2.1, sn, sd]
  rw [← tight_post, ← hpad]
  exact ⟨vp, vs, pr, ps, pd, sr, ss, sd, pn, sn, by rw [e1, e2], by rw [e3, e4], pdw, psw, pg, sg⟩

end stride_core
