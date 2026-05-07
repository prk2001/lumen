//! # lumen-perf
//!
//! Performance & hardware: GPU/CPU scheduling, memory, telemetry.
//!
//! This crate hosts the GPU compute scaffold built on `wgpu`. The first
//! reference kernel is brightness/contrast on RGBA-float frames; later
//! Phase-3 effects plug into the same [`GpuContext`].
//!
//! Status: scaffolding (Cat 27 — Performance & Hardware).
//! See `docs/PLAN.md` for the wider implementation roadmap.

#![forbid(unsafe_op_in_unsafe_fn)]

use std::borrow::Cow;
use std::sync::Arc;

use lumen_core::{Error, Frame, PixelData, Result};
use parking_lot::Mutex;
use tracing::{debug, instrument};

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// WGSL source for the brightness/contrast compute kernel.
///
/// Mirrors `lumen-fx-exposure::brightness_contrast`:
/// ```text
/// out = (in - 0.5) * contrast + 0.5 + brightness
/// ```
/// applied per RGB channel; alpha is preserved. RGBA pixels are packed as
/// `vec4<f32>` and bound as a storage buffer (read-write, in-place).
const BRIGHTNESS_CONTRAST_WGSL: &str = r#"
struct Params {
    width: u32,
    height: u32,
    brightness: f32,
    contrast: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pixels: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let idx: u32 = gid.y * params.width + gid.x;
    let p = pixels[idx];
    let rgb = (p.xyz - vec3<f32>(0.5)) * params.contrast
        + vec3<f32>(0.5) + vec3<f32>(params.brightness);
    let clamped = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    pixels[idx] = vec4<f32>(clamped, p.w);
}
"#;

/// CPU-side mirror of the WGSL `Params` uniform. `repr(C)` so we can
/// `bytemuck::cast_slice` directly into a uniform buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct BrightnessContrastParams {
    width: u32,
    height: u32,
    brightness: f32,
    contrast: f32,
}

// SAFETY: plain old data — four scalars, no padding, repr(C).
unsafe impl bytemuck::Zeroable for BrightnessContrastParams {}
unsafe impl bytemuck::Pod for BrightnessContrastParams {}

/// Lazily-initialized brightness/contrast pipeline + bind-group layout.
struct BrightnessContrastPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// GPU compute context — owns a `wgpu` device + queue and caches compiled
/// pipelines for the kernels in this crate.
///
/// Construct with [`try_new`]; pass by reference to kernel entry points
/// like [`run_brightness_contrast`].
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Cached pipeline for the brightness/contrast kernel, lazily built on
    /// first use.
    bc_pipeline: Mutex<Option<Arc<BrightnessContrastPipeline>>>,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext").finish_non_exhaustive()
    }
}

/// Try to construct a [`GpuContext`] backed by a default adapter.
///
/// Uses `wgpu::Backends::PRIMARY` and accepts the `Fallback` adapter type,
/// so software emulation (e.g. `lavapipe`/CPU backends in CI) is fine.
/// Returns `Err(Error::Other(...))` if no adapter is available — callers
/// (including tests) should treat that as "skip the GPU path."
pub async fn try_new() -> Result<GpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok_or_else(|| Error::Other("no GPU adapter available".to_string()))?;

    debug!(adapter = ?adapter.get_info(), "lumen-perf: selected GPU adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("lumen-perf device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|e| Error::Other(format!("wgpu: device request failed: {e}")))?;

    Ok(GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        bc_pipeline: Mutex::new(None),
    })
}

impl GpuContext {
    /// Returns the cached brightness/contrast pipeline, building it on the
    /// first call.
    fn brightness_contrast_pipeline(&self) -> Arc<BrightnessContrastPipeline> {
        let mut slot = self.bc_pipeline.lock();
        if let Some(p) = slot.as_ref() {
            return Arc::clone(p);
        }

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brightness_contrast.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BRIGHTNESS_CONTRAST_WGSL)),
        });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("brightness_contrast.bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("brightness_contrast.pll"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("brightness_contrast.pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let built = Arc::new(BrightnessContrastPipeline {
            pipeline,
            bind_group_layout,
        });
        *slot = Some(Arc::clone(&built));
        built
    }
}

/// Apply brightness/contrast on the GPU. Lifts the input via
/// [`Frame::into_rgba_f32_linear`] and returns a new frame in the same
/// color space, sized identically to the input.
///
/// This is a sync wrapper — it drives the async wgpu work via `pollster`.
#[instrument(skip(ctx, frame), fields(w = frame.width, h = frame.height))]
pub fn run_brightness_contrast(
    ctx: &GpuContext,
    frame: Frame,
    brightness: f32,
    contrast: f32,
) -> Result<Frame> {
    pollster::block_on(run_brightness_contrast_async(ctx, frame, brightness, contrast))
}

async fn run_brightness_contrast_async(
    ctx: &GpuContext,
    frame: Frame,
    brightness: f32,
    contrast: f32,
) -> Result<Frame> {
    let frame = frame.into_rgba_f32_linear();
    let width = frame.width;
    let height = frame.height;

    if width == 0 || height == 0 {
        return Ok(frame);
    }

    let f32_data: &[f32] = frame
        .as_f32()
        .expect("RgbaF32 after into_rgba_f32_linear");
    let byte_len = std::mem::size_of_val(f32_data) as wgpu::BufferAddress;

    let pipeline = ctx.brightness_contrast_pipeline();

    let params = BrightnessContrastParams {
        width,
        height,
        brightness,
        contrast,
    };

    let device = &ctx.device;
    let queue = &ctx.queue;

    // Uniform buffer with the kernel parameters.
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brightness_contrast.params"),
        size: std::mem::size_of::<BrightnessContrastParams>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    // Storage buffer with the pixel data, also used as a copy source for
    // readback.
    let pixel_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brightness_contrast.pixels"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&pixel_buf, 0, bytemuck::cast_slice(f32_data));

    // Staging buffer for read-back to the CPU.
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brightness_contrast.staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("brightness_contrast.bg"),
        layout: &pipeline.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pixel_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("brightness_contrast.encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("brightness_contrast.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = width.div_ceil(8);
        let wg_y = height.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    encoder.copy_buffer_to_buffer(&pixel_buf, 0, &staging_buf, 0, byte_len);

    queue.submit(Some(encoder.finish()));

    // Map the staging buffer for read.
    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel::<std::result::Result<(), wgpu::BufferAsyncError>>(1);
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    // Drive the GPU to flush in-flight work and the map callback.
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| Error::Other(format!("wgpu: map channel closed: {e}")))?
        .map_err(|e| Error::Other(format!("wgpu: map_async failed: {e}")))?;

    let mapped = slice.get_mapped_range();
    let out_pixels: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging_buf.unmap();

    Frame::new(
        width,
        height,
        PixelData::RgbaF32(out_pixels),
        frame.color_space,
        frame.pts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::ColorSpace;

    /// Try to acquire a GpuContext; if no adapter is available, print a
    /// skip message and return None so the test passes on GPU-less CI.
    fn ctx_or_skip(test_name: &str) -> Option<GpuContext> {
        match pollster::block_on(try_new()) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                eprintln!("[skip] {test_name}: no GPU adapter available ({e})");
                None
            }
        }
    }

    fn solid_frame(width: u32, height: u32, rgba: [f32; 4]) -> Frame {
        let n = (width as usize) * (height as usize);
        let mut data = Vec::with_capacity(n * 4);
        for _ in 0..n {
            data.extend_from_slice(&rgba);
        }
        Frame::new(
            width,
            height,
            PixelData::RgbaF32(data),
            ColorSpace::LinearSRgb,
            None,
        )
        .unwrap()
    }

    #[test]
    fn identity_passthrough() {
        let Some(ctx) = ctx_or_skip("identity_passthrough") else { return };

        // Mix of values across the channels to catch swizzle bugs.
        let input = solid_frame(16, 12, [0.1, 0.5, 0.9, 0.75]);
        let original = input.clone();
        let out = run_brightness_contrast(&ctx, input, 0.0, 1.0).unwrap();

        assert_eq!(out.width, 16);
        assert_eq!(out.height, 12);

        let in_px = original.as_f32().unwrap();
        let out_px = out.as_f32().unwrap();
        assert_eq!(in_px.len(), out_px.len());
        for (i, (a, b)) in in_px.iter().zip(out_px.iter()).enumerate() {
            // Alpha must match exactly; RGB within 1e-5.
            if i % 4 == 3 {
                assert!((a - b).abs() <= 1e-6, "alpha mismatch at {i}: {a} vs {b}");
            } else {
                assert!(
                    (a - b).abs() <= 1e-5,
                    "rgb mismatch at component {i}: {a} vs {b}",
                );
            }
        }
    }

    #[test]
    fn brightness_plus_one_clamps_white() {
        let Some(ctx) = ctx_or_skip("brightness_plus_one_clamps_white") else { return };

        let input = solid_frame(8, 8, [0.1, 0.2, 0.3, 0.4]);
        let out = run_brightness_contrast(&ctx, input, 1.0, 1.0).unwrap();

        let px = out.as_f32().unwrap();
        for chunk in px.chunks_exact(4) {
            assert!((chunk[0] - 1.0).abs() <= 1e-5, "R not clamped to 1.0: {}", chunk[0]);
            assert!((chunk[1] - 1.0).abs() <= 1e-5, "G not clamped to 1.0: {}", chunk[1]);
            assert!((chunk[2] - 1.0).abs() <= 1e-5, "B not clamped to 1.0: {}", chunk[2]);
            // Alpha preserved.
            assert!((chunk[3] - 0.4).abs() <= 1e-6, "alpha changed: {}", chunk[3]);
        }
    }

    #[test]
    fn contrast_zero_collapses_to_mid() {
        let Some(ctx) = ctx_or_skip("contrast_zero_collapses_to_mid") else { return };

        let input = solid_frame(4, 4, [0.5, 0.5, 0.5, 1.0]);
        let out = run_brightness_contrast(&ctx, input, 0.0, 0.0).unwrap();

        let px = out.as_f32().unwrap();
        for chunk in px.chunks_exact(4) {
            assert!((chunk[0] - 0.5).abs() <= 1e-5, "R: {}", chunk[0]);
            assert!((chunk[1] - 0.5).abs() <= 1e-5, "G: {}", chunk[1]);
            assert!((chunk[2] - 0.5).abs() <= 1e-5, "B: {}", chunk[2]);
            assert!((chunk[3] - 1.0).abs() <= 1e-6, "A: {}", chunk[3]);
        }
    }
}
