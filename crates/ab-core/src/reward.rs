//! Exact v3.3 reward mathematics (calibrated Pilot, Nov 2026).
//!
//! Mirrors `alphabuilder/src/logic/selfplay/reward.py` byte-for-byte in
//! behaviour. Constants are normative; do not replace them with v3.1/v3.2
//! values (`tanh`, `0.80 / 0.16 / 2.0`).

use serde::Serialize;

pub const COMPLIANCE_BASE: f64 = 1.15;
pub const COMPLIANCE_SLOPE: f64 = 0.35;
pub const COMPLIANCE_MIN: f64 = -0.60;
pub const COMPLIANCE_MAX: f64 = 0.90;

pub const VOLUME_REFERENCE: f64 = 0.10;
pub const VOLUME_SENSITIVITY: f64 = 2.50;
pub const VOLUME_BONUS_MAX: f64 = 0.30;
pub const VOLUME_PENALTY_MAX: f64 = 0.60;

pub const DISPLACEMENT_LIMIT_RAW: f64 = 128_000.0;
pub const REAL_SCALE_FACTOR: f64 = 500_000.0;

pub const PENALTY_PER_ISLAND: f64 = 0.02;
pub const PENALTY_PER_LOOSE_FRAC: f64 = 0.10;
pub const ISLAND_PENALTY_MAX: f64 = 0.30;

pub const DISPLACEMENT_PENALTY_START: f64 = 0.70;
pub const DISPLACEMENT_PENALTY_MAX: f64 = 0.50;

#[inline]
fn clip(value: f64, lo: f64, hi: f64) -> f64 {
    value.clamp(lo, hi)
}

/// `S_comp(C)` with the calibrated log10 mapping.
pub fn calculate_compliance_score(compliance: f64) -> f64 {
    if compliance <= 0.0 {
        return COMPLIANCE_MAX;
    }
    let log_c = compliance.max(1.0).log10();
    let score = COMPLIANCE_BASE - COMPLIANCE_SLOPE * (log_c - 1.0);
    clip(score, COMPLIANCE_MIN, COMPLIANCE_MAX)
}

/// `B_vol(V)`, always inversely proportional to volume.
pub fn calculate_volume_bonus(vol_frac: f64) -> f64 {
    let adjustment = (VOLUME_REFERENCE - vol_frac) * VOLUME_SENSITIVITY;
    clip(adjustment, -VOLUME_PENALTY_MAX, VOLUME_BONUS_MAX)
}

/// Pure physics score `S_fem = clip(S_comp + B_vol, -1, 1)`.
pub fn calculate_fem_score(compliance: f64, vol_frac: f64) -> f64 {
    clip(
        calculate_compliance_score(compliance) + calculate_volume_bonus(vol_frac),
        -1.0,
        1.0,
    )
}

/// Smooth collapse warning penalty above 70% of the raw displacement limit.
pub fn calculate_displacement_penalty(max_displacement: Option<f64>) -> f64 {
    match max_displacement {
        None => 0.0,
        Some(u) => {
            let ratio = u / DISPLACEMENT_LIMIT_RAW;
            if ratio <= DISPLACEMENT_PENALTY_START {
                0.0
            } else {
                DISPLACEMENT_PENALTY_MAX
                    * clip(
                        (ratio - DISPLACEMENT_PENALTY_START) / (1.0 - DISPLACEMENT_PENALTY_START),
                        0.0,
                        1.0,
                    )
            }
        }
    }
}

/// Game reward used by MCTS (invalid or collapsed states are -1.0).
pub fn calculate_reward(
    compliance: f64,
    vol_frac: f64,
    is_valid: bool,
    max_displacement: Option<f64>,
) -> f64 {
    if !is_valid {
        return -1.0;
    }
    if let Some(u) = max_displacement {
        if u > DISPLACEMENT_LIMIT_RAW {
            return -1.0;
        }
    }
    let score = calculate_fem_score(compliance, vol_frac);
    let penalty = calculate_displacement_penalty(max_displacement);
    if penalty > 0.0 {
        clip(score - penalty, -0.95, 1.0)
    } else {
        score
    }
}

/// Connectivity/island penalty `P_ilhas`.
pub fn calculate_island_penalty(n_islands: usize, loose_voxels: usize, total_voxels: usize) -> f64 {
    if n_islands <= 1 || total_voxels == 0 {
        return 0.0;
    }
    let extra_islands = (n_islands - 1) as f64;
    let island_penalty = PENALTY_PER_ISLAND * extra_islands;
    let loose_fraction = loose_voxels as f64 / total_voxels as f64;
    let loose_penalty = PENALTY_PER_LOOSE_FRAC * loose_fraction;
    (island_penalty + loose_penalty).min(ISLAND_PENALTY_MAX)
}

/// Full decomposition exposed to the UI.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct RewardBreakdown {
    pub compliance_score: f64,
    pub volume_bonus: f64,
    pub fem_score: f64,
    pub displacement_penalty: f64,
    pub island_penalty: f64,
    pub total_reward: f64,
    pub is_collapsed: bool,
}

/// Compute the exact v3.3 breakdown used by the Inspector panel.
pub fn calculate_reward_breakdown(
    compliance: f64,
    vol_frac: f64,
    is_valid: bool,
    max_displacement: Option<f64>,
    island_penalty: f64,
) -> RewardBreakdown {
    let compliance_score = calculate_compliance_score(compliance);
    let volume_bonus = calculate_volume_bonus(vol_frac);
    let fem_score = clip(compliance_score + volume_bonus, -1.0, 1.0);
    let displacement_penalty = calculate_displacement_penalty(max_displacement);
    let is_collapsed = !is_valid
        || max_displacement
            .map(|u| u > DISPLACEMENT_LIMIT_RAW)
            .unwrap_or(false);
    let total_reward = if is_collapsed {
        -1.0
    } else {
        let base = calculate_reward(compliance, vol_frac, is_valid, max_displacement);
        clip(base - island_penalty, -1.0, 1.0)
    };
    RewardBreakdown {
        compliance_score,
        volume_bonus,
        fem_score,
        displacement_penalty,
        island_penalty,
        total_reward,
        is_collapsed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected:.12}, got {actual:.12} (diff {:.3e})",
            (actual - expected).abs()
        );
    }

    /// The eight normative vectors from spec Section 5.3.
    #[test]
    fn reward_vectors_v33_exact() {
        type Vector = (f64, f64, bool, f64, f64, f64, f64, f64, f64);
        let cases: [Vector; 8] = [
            (100.0, 0.10, true, 50.0, 0.8, 0.0, 0.8, 0.0, 0.8),
            (1000.0, 0.10, true, 500.0, 0.45, 0.0, 0.45, 0.0, 0.45),
            (1000.0, 0.15, true, 500.0, 0.45, -0.125, 0.325, 0.0, 0.325),
            (
                1000.0,
                0.08,
                true,
                100_000.0,
                0.45,
                0.05,
                0.5,
                0.135_416_666_666_666_67,
                0.364_583_333_333_333_3,
            ),
            (1000.0, 0.10, true, 130_000.0, 0.45, 0.0, 0.45, 0.5, -1.0),
            (1000.0, 0.10, false, 50.0, 0.45, 0.0, 0.45, 0.0, -1.0),
            (1_000_000.0, 0.10, true, 100.0, -0.6, 0.0, -0.6, 0.0, -0.6),
            (10.0, 0.05, true, 10.0, 0.9, 0.125, 1.0, 0.0, 1.0),
        ];
        for (i, (c, v, valid, u, sc, bv, sf, pd, r)) in cases.into_iter().enumerate() {
            assert_close(calculate_compliance_score(c), sc);
            assert_close(calculate_volume_bonus(v), bv);
            assert_close(calculate_fem_score(c, v), sf);
            assert_close(calculate_displacement_penalty(Some(u)), pd);
            assert_close(calculate_reward(c, v, valid, Some(u)), r);
            let breakdown = calculate_reward_breakdown(c, v, valid, Some(u), 0.0);
            assert_close(breakdown.total_reward, r);
            assert_eq!(breakdown.is_collapsed, r == -1.0);
            println!("vector {i} ok");
        }
    }

    #[test]
    fn island_penalty_matches_python() {
        assert_close(calculate_island_penalty(0, 0, 0), 0.0);
        assert_close(calculate_island_penalty(1, 10, 10), 0.0);
        assert_close(calculate_island_penalty(2, 0, 100), 0.02);
        assert_close(calculate_island_penalty(3, 20, 100), 0.06);
        assert_close(calculate_island_penalty(10, 100, 100), 0.28);
        assert_close(calculate_island_penalty(20, 100, 100), 0.30);
    }

    #[test]
    fn zero_displacement_is_treated_as_no_penalty() {
        assert_close(calculate_displacement_penalty(None), 0.0);
        assert_close(calculate_displacement_penalty(Some(0.0)), 0.0);
        assert_close(
            calculate_displacement_penalty(Some(DISPLACEMENT_LIMIT_RAW)),
            0.5,
        );
    }

    #[test]
    fn spec_constants_are_exported() {
        assert_close(COMPLIANCE_BASE, 1.15);
        assert_close(COMPLIANCE_SLOPE, 0.35);
        assert_close(VOLUME_SENSITIVITY, 2.5);
        assert_close(DISPLACEMENT_LIMIT_RAW, 128_000.0);
    }
}
