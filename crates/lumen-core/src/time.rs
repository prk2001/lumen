//! Frame timing primitives.
//!
//! Video work demands rational frame rates (24000/1001, etc.). Floating
//! point silently accumulates drift; we use [`Rational`] everywhere a
//! timestamp needs to round-trip.

use serde::{Deserialize, Serialize};

/// A signed rational number, used for frame rates and presentation
/// timestamps. Always stored in normalized form (gcd-reduced, positive
/// denominator).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

impl Rational {
    /// Common rates as named constants.
    pub const FPS_24: Rational = Rational { num: 24, den: 1 };
    pub const FPS_24000_1001: Rational = Rational { num: 24000, den: 1001 };
    pub const FPS_25: Rational = Rational { num: 25, den: 1 };
    pub const FPS_30: Rational = Rational { num: 30, den: 1 };
    pub const FPS_30000_1001: Rational = Rational { num: 30000, den: 1001 };
    pub const FPS_50: Rational = Rational { num: 50, den: 1 };
    pub const FPS_60: Rational = Rational { num: 60, den: 1 };
    pub const FPS_60000_1001: Rational = Rational { num: 60000, den: 1001 };
    pub const FPS_120: Rational = Rational { num: 120, den: 1 };

    /// Construct, normalizing to positive denominator and reducing by gcd.
    /// `den == 0` produces `0/1` rather than panicking — callers that care
    /// about division-by-zero must check input.
    pub fn new(num: i64, den: i64) -> Self {
        if den == 0 {
            return Rational { num: 0, den: 1 };
        }
        let (mut n, mut d) = if den < 0 { (-num, -den) } else { (num, den) };
        let g = gcd(n.unsigned_abs(), d.unsigned_abs()) as i64;
        if g > 1 {
            n /= g;
            d /= g;
        }
        Rational { num: n, den: d }
    }

    /// Convert to f64. Lossy.
    pub fn as_f64(&self) -> f64 { self.num as f64 / self.den as f64 }

    /// True if the value is a whole, positive number.
    pub fn is_whole(&self) -> bool { self.den == 1 && self.num >= 0 }

    /// Reciprocal. `0/1` if `self.num == 0`.
    pub fn recip(&self) -> Rational {
        if self.num == 0 {
            Rational { num: 0, den: 1 }
        } else {
            Rational::new(self.den, self.num)
        }
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// A presentation timestamp expressed in a given timebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Pts {
    pub timebase: Rational,
    pub value: i64,
}

impl Pts {
    pub const ZERO: Pts = Pts { timebase: Rational { num: 1, den: 1 }, value: 0 };

    pub fn new(timebase: Rational, value: i64) -> Self { Self { timebase, value } }

    /// Seconds as f64.
    pub fn seconds(&self) -> f64 {
        self.value as f64 * self.timebase.as_f64()
    }

    /// Frame index assuming the supplied frame rate.
    pub fn frame_index(&self, frame_rate: Rational) -> i64 {
        // value * timebase * frame_rate
        let numer = self.value as i128
            * self.timebase.num as i128
            * frame_rate.num as i128;
        let denom = self.timebase.den as i128 * frame_rate.den as i128;
        (numer / denom) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_normalizes() {
        let r = Rational::new(48000, 2002);
        assert_eq!(r, Rational { num: 24000, den: 1001 });
    }

    #[test]
    fn rational_handles_negative_den() {
        let r = Rational::new(1, -2);
        assert_eq!(r, Rational { num: -1, den: 2 });
    }

    #[test]
    fn rational_zero_den_does_not_panic() {
        let r = Rational::new(7, 0);
        assert_eq!(r, Rational { num: 0, den: 1 });
    }

    #[test]
    fn pts_seconds_and_frame_index() {
        // 10 ticks at 1/30 timebase = 1/3 second
        let p = Pts::new(Rational::new(1, 30), 10);
        assert!((p.seconds() - (10.0 / 30.0)).abs() < 1e-9);
        // Frame index at 30fps == 10
        assert_eq!(p.frame_index(Rational::FPS_30), 10);
        // At 60fps == 20
        assert_eq!(p.frame_index(Rational::FPS_60), 20);
    }
}
