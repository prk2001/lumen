//! Content hashing — BLAKE3 of file bytes.
//!
//! We use BLAKE3 throughout (fast, parallelizable, modern) and prefix
//! every hex digest with `"blake3:"` so the algorithm is self-describing
//! in JSON.

use std::path::Path;

use lumen_core::{Error, Result};

/// Hash a file's bytes with BLAKE3 and return `"blake3:<hex>"`.
pub fn hash_file<P: AsRef<Path>>(path: P) -> Result<String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|e| Error::Io(std::io::Error::new(e.kind(), format!("hash_file: {e}"))))?;
    Ok(hash_bytes(&bytes))
}

/// Hash bytes with BLAKE3 and return `"blake3:<hex>"`.
pub fn hash_bytes(bytes: &[u8]) -> String {
    let h = blake3::hash(bytes);
    format!("blake3:{}", h.to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_bytes_is_stable() {
        let a = hash_bytes(b"hello");
        let b = hash_bytes(b"hello");
        assert_eq!(a, b);
        assert!(a.starts_with("blake3:"));
        // Empty input has a known BLAKE3 digest.
        let empty = hash_bytes(b"");
        assert!(empty.starts_with("blake3:"));
    }
}
