//! Push constants and workgroup count of the stride-copy shader.
//!
//! `shaders/stride_copy.comp` indexes its two storage buffers in 32-bit
//! arithmetic. [`plan`] accepts a copy only when both bindings fit the
//! device's `maxStorageBufferRange` and the dispatch fits its
//! `maxComputeWorkGroupCount[0]`. `verify/` extracts this file to Lean with
//! Charon and Aeneas and proves that every invocation of an accepted dispatch
//! reads and writes inside its bindings without wrapping, that the dispatch
//! covers every word of every row and that no two invocations write the same
//! word.

/// Local workgroup size declared in `stride_copy.comp`.
pub(crate) const LOCAL_SIZE_X: u32 = 64;

/// One stride-copy dispatch: the shader's push constants, the word counts of
/// its two bindings and its workgroup count.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StrideCopy {
    /// Words copied per row.
    pub(crate) row_uints: u32,
    /// Words from the start of one source row to the start of the next.
    pub(crate) src_stride_uints: u32,
    /// Words from the start of one destination row to the start of the next.
    pub(crate) dst_stride_uints: u32,
    /// Number of rows.
    pub(crate) n_rows: u32,
    /// Words bound at binding 0, `n_rows * src_stride_uints`.
    pub(crate) src_words: u32,
    /// Words bound at binding 1, `n_rows * dst_stride_uints`.
    pub(crate) dst_words: u32,
    /// Workgroups along x, `ceil(row_uints * n_rows / LOCAL_SIZE_X)`.
    pub(crate) groups_x: u32,
}

/// Dispatch copying `n_rows` rows of `row_uints` words between buffers with
/// the given row strides, or `None` when a row is empty, a row is longer than
/// either stride, a binding exceeds `max_range_bytes` or the dispatch needs
/// more than `max_groups` workgroups.
pub(crate) fn plan(
    row_uints: u64,
    src_stride_uints: u64,
    dst_stride_uints: u64,
    n_rows: u64,
    max_range_bytes: u32,
    max_groups: u32,
) -> Option<StrideCopy> {
    let max_words = (max_range_bytes / 4) as u64;
    if row_uints == 0 || n_rows == 0 {
        return None;
    }
    if row_uints > src_stride_uints || row_uints > dst_stride_uints {
        return None;
    }
    if n_rows > max_words / src_stride_uints || n_rows > max_words / dst_stride_uints {
        return None;
    }
    let src_words = n_rows * src_stride_uints;
    let dst_words = n_rows * dst_stride_uints;
    let total = row_uints * n_rows;
    // Written out so that Aeneas can translate it.
    #[allow(clippy::manual_div_ceil)]
    let groups_x = (total + (LOCAL_SIZE_X as u64 - 1)) / LOCAL_SIZE_X as u64;
    if groups_x > max_groups as u64 {
        return None;
    }
    Some(StrideCopy {
        row_uints: row_uints as u32,
        src_stride_uints: src_stride_uints as u32,
        dst_stride_uints: dst_stride_uints as u32,
        n_rows: n_rows as u32,
        src_words: src_words as u32,
        dst_words: dst_words as u32,
        groups_x: groups_x as u32,
    })
}

/// Pad and strip dispatches of a real transform whose innermost axis has
/// `innermost` reals of `elem_uints` words each, over `n_rows` rows.
///
/// The pad copies tight rows of `innermost` reals into rows padded to
/// `2 * (innermost / 2 + 1)` reals. The strip copies them back.
pub(crate) fn real_transform(
    innermost: u32,
    n_rows: u64,
    elem_uints: u32,
    max_range_bytes: u32,
    max_groups: u32,
) -> Option<(StrideCopy, StrideCopy)> {
    let tight = innermost as u64 * elem_uints as u64;
    let padded = 2 * (innermost as u64 / 2 + 1) * elem_uints as u64;
    // A match in place of `?`, which Aeneas does not model on `Option`.
    match (
        plan(tight, tight, padded, n_rows, max_range_bytes, max_groups),
        plan(tight, padded, tight, n_rows, max_range_bytes, max_groups),
    ) {
        (Some(pad), Some(strip)) => Some((pad, strip)),
        _ => None,
    }
}

/// Source and destination word indices of invocation `tid`, computed in the
/// shader's `u32` arithmetic, or `None` for an invocation that returns early.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn invocation(p: &StrideCopy, tid: u32) -> Option<(u32, u32)> {
    let total = p.row_uints * p.n_rows;
    if tid >= total {
        return None;
    }
    let row = tid / p.row_uints;
    let col = tid - row * p.row_uints;
    Some((
        row * p.src_stride_uints + col,
        row * p.dst_stride_uints + col,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs every invocation of `p` on the CPU.
    fn run(p: &StrideCopy, src: &[u32], dst: &mut [u32]) {
        for tid in 0..p.groups_x * LOCAL_SIZE_X {
            if let Some((s, d)) = invocation(p, tid) {
                dst[d as usize] = src[s as usize];
            }
        }
    }

    #[test]
    fn pad_then_strip_restores_every_row() {
        for innermost in [1u32, 2, 7, 64, 255] {
            for elem_uints in [1u32, 2] {
                for n_rows in [1u64, 3, 17] {
                    let (pad, strip) =
                        real_transform(innermost, n_rows, elem_uints, u32::MAX, 65535).unwrap();
                    let src: Vec<u32> = (0..pad.src_words).collect();
                    let mut padded = vec![u32::MAX; pad.dst_words as usize];
                    run(&pad, &src, &mut padded);
                    assert_eq!(strip.src_words, pad.dst_words);
                    let mut back = vec![0; strip.dst_words as usize];
                    run(&strip, &padded, &mut back);
                    assert_eq!(back, src, "{innermost} {elem_uints} {n_rows}");
                }
            }
        }
    }

    #[test]
    fn limits_reject_oversized_dispatches() {
        // llvmpipe: 128 MiB bindings, 65535 workgroups.
        let most = 65535 * 64;
        assert!(plan(most, most, most, 1, 1 << 27, 65535).is_some());
        assert!(plan(most + 1, most + 1, most + 1, 1, 1 << 27, 65535).is_none());
        assert!(plan(1 << 25, 1 << 25, 1 << 25, 1, 1 << 27, 1 << 20).is_some());
        assert!(plan(1 << 25, 1 << 25, (1 << 25) + 1, 1, 1 << 27, 1 << 20).is_none());
        assert!(plan(4, 3, 4, 1, u32::MAX, 1).is_none());
        assert!(plan(0, 1, 1, 1, u32::MAX, 1).is_none());
        assert!(real_transform(u32::MAX, 1, 2, u32::MAX, u32::MAX).is_none());
    }
}
