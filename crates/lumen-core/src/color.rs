//! Color space identifiers and transfer-function helpers.
//!
//! Lumen's working space is **scene-linear ACEScg float32**. Every
//! decode converts into linear (where possible); every encode applies a
//! display transform on the way out. This module names the spaces we
//! understand. Heavy lifting (matrices, OCIO configs) lives in
//! `lumen-color`.
//!
//! Source spec: Cat 5 (Color Science & Grading) and Cat 4 (Exposure,
//! Tone & Dynamic Range). Round 1 of this module covers identification
//! only — bidirectional matrix conversion is in `lumen-color`.

use serde::{Deserialize, Serialize};

/// A named color space.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    /// sRGB primaries with the sRGB transfer function (gamma-encoded).
    #[default]
    SRgb,
    /// sRGB primaries, linear.
    LinearSRgb,
    /// Rec.709 primaries with the BT.1886 / sRGB-like transfer.
    Rec709,
    /// Rec.709 primaries, linear.
    LinearRec709,
    /// Rec.2020 primaries with PQ (SMPTE ST 2084) transfer.
    Rec2020Pq,
    /// Rec.2020 primaries with HLG transfer.
    Rec2020Hlg,
    /// Rec.2020 primaries, linear.
    LinearRec2020,
    /// ACEScg working space (AP1 primaries, linear).
    AcesCg,
    /// ACES2065-1 archival space (AP0 primaries, linear).
    Aces2065,
    /// DCI-P3 with the DCI gamma 2.6 transfer.
    DciP3,
    /// Display P3 with the sRGB transfer.
    DisplayP3,
    /// Linear DCI-P3.
    LinearDciP3,
    /// ARRI LogC v3.
    ArriLogC3,
    /// Sony S-Log3.
    SLog3,
    /// Apple Log.
    AppleLog,
    /// Vendor-specific / unknown — caller carries an opaque identifier.
    Custom(String),
}

impl ColorSpace {
    /// True if the space is already in scene-linear math, i.e. no
    /// transfer function is applied.
    pub fn is_linear(&self) -> bool {
        matches!(
            self,
            ColorSpace::LinearSRgb
                | ColorSpace::LinearRec709
                | ColorSpace::LinearRec2020
                | ColorSpace::AcesCg
                | ColorSpace::Aces2065
                | ColorSpace::LinearDciP3
        )
    }

    /// Approximate display peak luminance (cd/m²) for tone-map planning.
    /// `None` for scene-linear / log spaces where it's not meaningful.
    pub fn nominal_peak_nits(&self) -> Option<f32> {
        Some(match self {
            ColorSpace::SRgb | ColorSpace::DisplayP3 | ColorSpace::Rec709 => 100.0,
            ColorSpace::DciP3 => 48.0,
            ColorSpace::Rec2020Pq => 10_000.0,
            ColorSpace::Rec2020Hlg => 1_000.0,
            _ => return None,
        })
    }

    /// String slug used in JSON / config files.
    pub fn slug(&self) -> &str {
        match self {
            ColorSpace::SRgb => "srgb",
            ColorSpace::LinearSRgb => "linear_srgb",
            ColorSpace::Rec709 => "rec709",
            ColorSpace::LinearRec709 => "linear_rec709",
            ColorSpace::Rec2020Pq => "rec2020_pq",
            ColorSpace::Rec2020Hlg => "rec2020_hlg",
            ColorSpace::LinearRec2020 => "linear_rec2020",
            ColorSpace::AcesCg => "aces_cg",
            ColorSpace::Aces2065 => "aces2065",
            ColorSpace::DciP3 => "dci_p3",
            ColorSpace::DisplayP3 => "display_p3",
            ColorSpace::LinearDciP3 => "linear_dci_p3",
            ColorSpace::ArriLogC3 => "arri_log_c3",
            ColorSpace::SLog3 => "s_log3",
            ColorSpace::AppleLog => "apple_log",
            ColorSpace::Custom(s) => s.as_str(),
        }
    }
}

/// sRGB transfer function (gamma-encoded → linear).
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB transfer function (linear → gamma-encoded).
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_round_trip() {
        for v in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let lin = srgb_to_linear(v);
            let back = linear_to_srgb(lin);
            assert!((back - v).abs() < 1e-5, "round trip failed at {v}: got {back}");
        }
    }

    #[test]
    fn nominal_peak_nits_known() {
        assert_eq!(ColorSpace::Rec2020Pq.nominal_peak_nits(), Some(10_000.0));
        assert_eq!(ColorSpace::SRgb.nominal_peak_nits(), Some(100.0));
        assert_eq!(ColorSpace::AcesCg.nominal_peak_nits(), None);
    }
}
