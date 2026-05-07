//! Session — load an ONNX model with [`tract`] and run inference.
//!
//! tract is pure Rust, so unlike `ort` there is no system library or
//! dynamic loader to wrangle: a single `cargo build` produces a binary
//! that runs anywhere the workspace itself runs. The trade-off is that
//! tract supports a subset of the ONNX op set — large enough for the
//! vision models Lumen ships in Phase 2 (denoise, upscale, detection,
//! segmentation), but not 1:1 with ONNX-Runtime.
//!
//! Loading flow:
//!
//! ```text
//! tract_onnx::onnx()
//!     .model_for_path(path)?       // -> InferenceModel
//!     .into_optimized()?           // -> TypedModel (constant-folded, simplified)
//!     .into_runnable()?            // -> TypedRunnableModel (executable plan)
//! ```

use std::path::{Path, PathBuf};

use tract_onnx::prelude::*;

use crate::error::{AiError, Result};
use crate::tensor::ImageTensor;

/// A loaded, optimized, runnable ONNX model.
///
/// Wraps tract's [`TypedRunnableModel`] so callers don't need to depend
/// on tract directly. Construct via [`load_session`] or
/// [`load_session_from_bytes`].
pub struct Session {
    inner: TypedRunnableModel<TypedModel>,
    source_path: Option<PathBuf>,
}

impl Session {
    /// Path the model was loaded from, if any. `None` for sessions
    /// constructed from in-memory bytes.
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// Number of input slots the model expects.
    pub fn input_count(&self) -> usize {
        self.inner.model().inputs.len()
    }

    /// Number of output slots the model produces.
    pub fn output_count(&self) -> usize {
        self.inner.model().outputs.len()
    }

    /// Borrow the underlying tract runnable model. Effects that need
    /// fine-grained control (multiple inputs, custom output extraction)
    /// can use this; the [`run`] helper covers the common path.
    pub fn inner(&self) -> &TypedRunnableModel<TypedModel> {
        &self.inner
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("source_path", &self.source_path)
            .field("input_count", &self.input_count())
            .field("output_count", &self.output_count())
            .finish()
    }
}

/// Load and optimize an ONNX model from disk into a [`Session`].
pub fn load_session<P: AsRef<Path>>(path: P) -> Result<Session> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(AiError::ModelNotFound(path.to_path_buf()));
    }
    tracing::debug!(path = %path.display(), "loading ONNX session via tract");

    let inner = tract_onnx::onnx()
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;

    Ok(Session {
        inner,
        source_path: Some(path.to_path_buf()),
    })
}

/// Load and optimize an ONNX model from a byte slice (e.g. via
/// `include_bytes!`).
pub fn load_session_from_bytes(bytes: &[u8]) -> Result<Session> {
    let mut cursor = std::io::Cursor::new(bytes);
    let inner = tract_onnx::onnx()
        .model_for_read(&mut cursor)?
        .into_optimized()?
        .into_runnable()?;
    Ok(Session {
        inner,
        source_path: None,
    })
}

/// Bytes of a minimal ONNX model containing a single Identity node
/// over a `1×3×4×4` float tensor.
///
/// Generated once with the helper script in
/// `crates/lumen-ai/test-models/README.md` (Python `onnx` package) and
/// checked in so the test suite runs offline. The choice of a
/// pre-generated, checked-in file (rather than constructing the
/// protobuf bytes by hand) keeps the test deterministic and the source
/// reviewable without asking readers to grok the ONNX wire format.
const IDENTITY_ONNX: &[u8] = include_bytes!("../test-models/identity.onnx");

/// Run [`Session::inner`] on a single CHW float tensor and return the
/// first output as another [`ImageTensor`].
///
/// Assumes a single-input single-output graph whose first output is a
/// 4-D float tensor. Effects with more exotic graphs should use
/// [`Session::inner`] directly.
pub fn run(session: &Session, input: &ImageTensor) -> Result<ImageTensor> {
    // Build a tract Tensor from a borrow of the input data — we copy
    // into an ndarray Array4 so tract owns its buffer.
    let array = tract_ndarray::Array4::<f32>::from_shape_vec(
        (input.shape[0], input.shape[1], input.shape[2], input.shape[3]),
        input.data.clone(),
    )
    .map_err(|e| AiError::Shape(format!("failed to build ndarray from input: {e}")))?;

    let tensor: Tensor = array.into();
    let outputs = session.inner.run(tvec!(tensor.into()))?;

    if outputs.is_empty() {
        return Err(AiError::Shape("session produced no outputs".to_string()));
    }
    let out = &outputs[0];
    let view = out.to_array_view::<f32>()?;
    let shape_vec = view.shape().to_vec();
    if shape_vec.len() != 4 {
        return Err(AiError::Shape(format!(
            "expected 4-D output, got {}-D shape {:?}",
            shape_vec.len(),
            shape_vec
        )));
    }
    let data: Vec<f32> = view.iter().copied().collect();
    Ok(ImageTensor {
        data,
        shape: [shape_vec[0], shape_vec[1], shape_vec[2], shape_vec[3]],
    })
}

/// End-to-end smoke test that exercises model load, optimize, and run.
///
/// Loads the embedded `identity.onnx` model, runs it on a
/// pseudo-random `1×3×4×4` float tensor, and asserts the output equals
/// the input within `1e-6`. Used both as a unit test and as a runtime
/// self-check so deployments can detect a broken inference stack
/// before hitting it from a real effect.
pub fn run_identity_check() -> Result<()> {
    let session = load_session_from_bytes(IDENTITY_ONNX)?;

    // Deterministic pseudo-random fill so failures are reproducible.
    // (We don't need crypto-quality randomness here; a linear
    // congruential generator is more than enough to detect any
    // permutation/byte-order regression.)
    let mut state: u32 = 0x9E37_79B9;
    let mut data = vec![0.0f32; 3 * 4 * 4];
    for slot in data.iter_mut() {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        // Map to [0, 1) without bias toward 0.
        *slot = ((state >> 8) & 0x00FF_FFFF) as f32 / (1u32 << 24) as f32;
    }
    let input = ImageTensor { data: data.clone(), shape: [1, 3, 4, 4] };

    let output = run(&session, &input)?;

    if output.shape != input.shape {
        return Err(AiError::Shape(format!(
            "identity output shape {:?} != input shape {:?}",
            output.shape, input.shape
        )));
    }
    if output.data.len() != input.data.len() {
        return Err(AiError::Shape(format!(
            "identity output length {} != input length {}",
            output.data.len(),
            input.data.len()
        )));
    }
    for (i, (&a, &b)) in output.data.iter().zip(input.data.iter()).enumerate() {
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
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-model.onnx");
        std::fs::write(&path, b"this is definitely not an onnx protobuf!!!").unwrap();
        let err = load_session(&path).expect_err("expected load to fail");
        // We don't care which variant — just that it failed and that
        // the error wasn't a missing-file error.
        match err {
            AiError::Tract(_) | AiError::Other(_) => {}
            AiError::ModelNotFound(_) => {
                panic!("file exists but tract claims it's missing")
            }
            other => panic!("unexpected error kind: {other:?}"),
        }
    }

    #[test]
    fn missing_model_errors() {
        let err = load_session("/definitely/not/a/real/path.onnx").unwrap_err();
        assert!(matches!(err, AiError::ModelNotFound(_)));
    }

    #[test]
    fn loaded_session_reports_io_counts() {
        let session = load_session_from_bytes(IDENTITY_ONNX).unwrap();
        assert_eq!(session.input_count(), 1);
        assert_eq!(session.output_count(), 1);
    }
}
