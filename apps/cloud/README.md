# Cloud / SaaS deployment

Deployment manifests and runbooks for hosting `lumen-server` (Axum HTTP
binary in `crates/lumen-server`) as a managed service.

**Status:** placeholder. Server-side rendering and project sharing
arrive in Phase 5–6.

## Planned shape

- `Dockerfile` — multi-stage Rust + ffmpeg + ONNX EPs base image
- `compose.yaml` — local dev with Postgres + object store
- `terraform/` — production infra (likely AWS Fargate + S3 + RDS)
- `k8s/` — Kubernetes manifests for self-hosters
- `runbooks/` — incident playbooks
