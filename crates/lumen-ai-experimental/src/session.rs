//! Session — load an ONNX model and run inference.
//!
//! On macOS we register the CoreML execution provider before falling
//! back to CPU. On other platforms we use ort's default routing, which
//! today resolves to the CPU EP. Hardware-accelerated EPs (CUDA,
//! DirectML, …) will be added in later phases as the targeted models
//! demand them.

use std::path::{Path, PathBuf};

use ndarray::Array4;
use ort::session::builder::SessionBuilder;
use ort::session::Session as OrtSession;
use ort::value::Tensor;

use crate::error::{AiError, Result};

/// A loaded ONNX model ready to run.
///
/// Wraps [`ort::session::Session`] so we can attach Lumen-specific
/// metadata (the path it was loaded from, the EPs it ended up using)
/// without exposing `ort` types in our public signatures.
pub struct Session {
    inner: OrtSession,
    source_path: Option<PathBuf>,
}

impl Session {
    /// Borrow the underlying [`ort::session::Session`]. Effects that
    /// need fine-grained control over inputs/outputs use this; the
    /// helpers on [`Session`] cover the common paths.
    pub fn ort(&self) -> &OrtSession {
        &self.inner
    }

    /// Mutable access to the underlying ort session — required to call
    /// `run` in ort 2.0 since it takes `&mut self`.
    pub fn ort_mut(&mut self) -> &mut OrtSession {
        &mut self.inner
    }

    /// Path the model was loaded from, if any. `None` for sessions
    /// constructed from in-memory bytes.
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }
}

/// Build a [`SessionBuilder`] with Lumen's standard EP order:
/// CoreML on macOS, then CPU fallback.
fn configure_builder() -> Result<SessionBuilder> {
    let builder = SessionBuilder::new()?;

    #[cfg(target_os = "macos")]
    {
        use ort::execution_providers::CoreMLExecutionProvider;
        // `error_on_failure(false)` so that if CoreML can't load the
        // graph (some ops aren't supported) we silently fall through
        // to CPU instead of failing.
        let coreml = CoreMLExecutionProvider::default();
        // `with_execution_providers` accepts a slice of EP dispatchers.
        // Order matters: ort tries them top-down.
        let builder = builder.with_execution_providers([coreml.build().error_on_failure(false)])?;
        return Ok(builder);
    }

    #[cfg(not(target_os = "macos"))]
    Ok(builder)
}

/// Load an ONNX model from disk into a [`Session`].
pub fn load_session(model_path: &Path) -> Result<Session> {
    if !model_path.exists() {
        return Err(AiError::ModelNotFound(model_path.to_path_buf()));
    }
    tracing::debug!(path = %model_path.display(), "loading ONNX session");

    let builder = configure_builder()?;
    let inner = builder.commit_from_file(model_path)?;

    Ok(Session {
        inner,
        source_path: Some(model_path.to_path_buf()),
    })
}

/// Load an ONNX model from a byte slice (e.g. `include_bytes!`).
pub fn load_session_from_bytes(bytes: &[u8]) -> Result<Session> {
    let builder = configure_builder()?;
    let inner = builder.commit_from_memory(bytes)?;
    Ok(Session {
        inner,
        source_path: None,
    })
}

/// Bytes of a minimal ONNX model containing a single Identity node.
///
/// Generated with the helper script in
/// `crates/lumen-ai/test-models/README.md` and checked in so the test
/// suite can run offline.
const IDENTITY_ONNX: &[u8] = include_bytes!("../test-models/identity.onnx");

/// End-to-end smoke test that exercises model load + run.
///
/// Loads a tiny in-memory ONNX model containing a single `Identity`
/// node operating on a `1×3×4×4` float tensor, runs it, and asserts
/// that the output equals the input. Used both as a unit test and as a
/// runtime self-check so deployments can detect a broken ort install
/// before hitting it from a real effect.
pub fn run_identity_check() -> Result<()> {
    let mut session = load_session_from_bytes(IDENTITY_ONNX)?;

    // 1×3×4×4 sequential floats — easy to spot if anything goes wrong.
    let input: Array4<f32> = Array4::from_shape_fn((1, 3, 4, 4), |(_, c, y, x)| {
        (c as f32) * 100.0 + (y as f32) * 10.0 + (x as f32)
    });
    let expected = input.clone();

    let input_tensor = Tensor::from_array(input)?;
    let outputs = session
        .ort_mut()
        .run(ort::inputs!["input" => input_tensor])?;

    // The graph names its output "output". Pull it back as &[f32].
    let (_shape, out_data) = outputs["output"].try_extract_tensor::<f32>()?;

    let expected_slice = expected.as_slice().expect("contiguous");
    if out_data.len() != expected_slice.len() {
        return Err(AiError::Shape(format!(
            "identity output length {} != input length {}",
            out_data.len(),
            expected_slice.len()
        )));
    }
    for (i, (&a, &b)) in out_data.iter().zip(expected_slice.iter()).enumerate() {
        if (a - b).abs() > 1e-6 {
            return Err(AiError::Other(format!(
                "identity mismatch at {i}: expected {b}, got {a}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_round_trip() {
        run_identity_check().expect("identity check should pass");
    }

    #[test]
    fn malformed_model_errors() {
        // Write 32 bytes of garbage to a temp file and try to load it.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-model.onnx");
        std::fs::write(&path, b"this is definitely not an onnx protobuf!!!").unwrap();
        let err = load_session(&path).expect_err("expected load to fail");
        // We don't care which variant — just that it failed.
        match err {
            AiError::Ort(_) | AiError::Other(_) => {}
            other => panic!("expected Ort/Other error, got {other:?}"),
        }
    }

    #[test]
    fn missing_model_errors() {
        let err = load_session(Path::new("/definitely/not/a/real/path.onnx")).unwrap_err();
        assert!(matches!(err, AiError::ModelNotFound(_)));
    }
}
