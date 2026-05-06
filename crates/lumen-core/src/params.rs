//! Effect parameters.
//!
//! Every effect declares a fixed list of [`ParamSpec`] entries describing
//! its inputs. At apply-time the host hands the effect a [`ParamValues`]
//! map keyed by parameter id. Both spec and values serialize round-trip
//! to JSON for project files.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{Error, Result};

/// Declarative metadata about a single parameter.
///
/// Defined in code by each effect — not loaded from JSON — so it carries
/// `&'static str` references and only needs `Serialize` for UI/API
/// introspection.
#[derive(Debug, Clone, Serialize)]
pub struct ParamSpec {
    /// Stable id used in project files (e.g. `"strength"`, `"radius"`).
    pub id: &'static str,
    /// Human-readable name shown in UI.
    pub display_name: &'static str,
    /// One-sentence help string.
    pub description: &'static str,
    /// Type and default.
    pub kind: ParamKind,
}

/// Parameter type plus default and bounds.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParamKind {
    Bool { default: bool },
    Int { default: i64, min: Option<i64>, max: Option<i64> },
    Float { default: f64, min: Option<f64>, max: Option<f64> },
    Choice { default: &'static str, options: &'static [&'static str] },
    String { default: &'static str },
}

/// Concrete value for a parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

/// A keyed bundle of parameter values, paired with the spec list at
/// validation time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParamValues {
    values: BTreeMap<String, ParamValue>,
}

impl ParamValues {
    pub fn new() -> Self { Self::default() }

    pub fn insert(&mut self, key: impl Into<String>, value: ParamValue) -> &mut Self {
        self.values.insert(key.into(), value);
        self
    }

    pub fn get(&self, key: &str) -> Option<&ParamValue> { self.values.get(key) }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            ParamValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn get_int(&self, key: &str) -> Option<i64> {
        match self.get(key)? {
            ParamValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            ParamValue::Float(f) => Some(*f),
            ParamValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            ParamValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Validate this set of values against a list of [`ParamSpec`]s.
    /// Missing values are filled in from defaults; unknown keys are
    /// reported as errors.
    pub fn validate_and_fill(&mut self, specs: &[ParamSpec]) -> Result<()> {
        // Reject unknown keys.
        for k in self.values.keys() {
            if !specs.iter().any(|s| s.id == k) {
                return Err(Error::InvalidParameter {
                    name: k.clone(),
                    reason: "unknown parameter".to_string(),
                });
            }
        }

        for spec in specs {
            match self.values.get(spec.id) {
                Some(value) => match (&spec.kind, value) {
                    (ParamKind::Bool { .. }, ParamValue::Bool(_))
                    | (ParamKind::String { .. }, ParamValue::String(_))
                    | (ParamKind::Choice { .. }, ParamValue::String(_)) => {}
                    (ParamKind::Int { min, max, .. }, ParamValue::Int(v)) => {
                        if let Some(lo) = min {
                            if v < lo {
                                return Err(Error::InvalidParameter {
                                    name: spec.id.to_string(),
                                    reason: format!("{v} below min {lo}"),
                                });
                            }
                        }
                        if let Some(hi) = max {
                            if v > hi {
                                return Err(Error::InvalidParameter {
                                    name: spec.id.to_string(),
                                    reason: format!("{v} above max {hi}"),
                                });
                            }
                        }
                    }
                    (ParamKind::Float { min, max, .. }, ParamValue::Float(v)) => {
                        if let Some(lo) = min {
                            if v < lo {
                                return Err(Error::InvalidParameter {
                                    name: spec.id.to_string(),
                                    reason: format!("{v} below min {lo}"),
                                });
                            }
                        }
                        if let Some(hi) = max {
                            if v > hi {
                                return Err(Error::InvalidParameter {
                                    name: spec.id.to_string(),
                                    reason: format!("{v} above max {hi}"),
                                });
                            }
                        }
                    }
                    (ParamKind::Float { .. }, ParamValue::Int(i)) => {
                        // Promote int to float silently.
                        self.values
                            .insert(spec.id.to_string(), ParamValue::Float(*i as f64));
                    }
                    (kind, value) => {
                        return Err(Error::InvalidParameter {
                            name: spec.id.to_string(),
                            reason: format!(
                                "type mismatch: spec={kind:?} value={value:?}"
                            ),
                        });
                    }
                },
                None => {
                    // Fill from default.
                    let v = match &spec.kind {
                        ParamKind::Bool { default } => ParamValue::Bool(*default),
                        ParamKind::Int { default, .. } => ParamValue::Int(*default),
                        ParamKind::Float { default, .. } => ParamValue::Float(*default),
                        ParamKind::Choice { default, .. } => {
                            ParamValue::String(default.to_string())
                        }
                        ParamKind::String { default } => {
                            ParamValue::String(default.to_string())
                        }
                    };
                    self.values.insert(spec.id.to_string(), v);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECS: &[ParamSpec] = &[
        ParamSpec {
            id: "strength",
            display_name: "Strength",
            description: "Effect strength",
            kind: ParamKind::Float { default: 0.5, min: Some(0.0), max: Some(1.0) },
        },
        ParamSpec {
            id: "preset",
            display_name: "Preset",
            description: "",
            kind: ParamKind::Choice { default: "auto", options: &["auto", "manual"] },
        },
    ];

    #[test]
    fn fills_defaults() {
        let mut pv = ParamValues::new();
        pv.validate_and_fill(SPECS).unwrap();
        assert_eq!(pv.get_float("strength"), Some(0.5));
        assert_eq!(pv.get_string("preset"), Some("auto"));
    }

    #[test]
    fn rejects_unknown_key() {
        let mut pv = ParamValues::new();
        pv.insert("nope", ParamValue::Bool(true));
        let r = pv.validate_and_fill(SPECS);
        assert!(matches!(r, Err(Error::InvalidParameter { .. })));
    }

    #[test]
    fn rejects_out_of_range() {
        let mut pv = ParamValues::new();
        pv.insert("strength", ParamValue::Float(2.0));
        let r = pv.validate_and_fill(SPECS);
        assert!(matches!(r, Err(Error::InvalidParameter { .. })));
    }

    #[test]
    fn promotes_int_to_float() {
        let mut pv = ParamValues::new();
        pv.insert("strength", ParamValue::Int(1));
        pv.validate_and_fill(SPECS).unwrap();
        assert_eq!(pv.get_float("strength"), Some(1.0));
    }
}
