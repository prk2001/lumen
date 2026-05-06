//! Effect registry.
//!
//! Effects register themselves at process start. The CLI, server, and
//! desktop app all build a single shared registry and look effects up
//! by id when executing a [`Graph`](crate::graph::Graph).

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::effect::{Effect, EffectRef};
use crate::error::{Error, Result};

/// Thread-safe registry of effects, keyed by stable id.
#[derive(Debug, Default)]
pub struct EffectRegistry {
    inner: RwLock<BTreeMap<String, EffectRef>>,
}

impl EffectRegistry {
    pub fn new() -> Self { Self::default() }

    /// Register an effect under its `metadata().id`. Returns `Err` if a
    /// different effect was already registered with the same id.
    pub fn register(&self, effect: EffectRef) -> Result<()> {
        let id = effect.metadata().id.to_string();
        let mut g = self.inner.write();
        if g.contains_key(&id) {
            return Err(Error::Other(format!("effect '{id}' already registered")));
        }
        g.insert(id, effect);
        Ok(())
    }

    /// Convenience: register a `T: Effect + 'static` directly.
    pub fn register_default<T: Effect + Default + 'static>(&self) -> Result<()> {
        self.register(Arc::new(T::default()))
    }

    pub fn get(&self, id: &str) -> Option<EffectRef> {
        self.inner.read().get(id).cloned()
    }

    /// All registered effect ids, sorted.
    pub fn ids(&self) -> Vec<String> {
        self.inner.read().keys().cloned().collect()
    }

    pub fn len(&self) -> usize { self.inner.read().len() }
    pub fn is_empty(&self) -> bool { self.inner.read().is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::effect::{Capabilities, Category, EffectMetadata};
    use crate::frame::Frame;
    use crate::params::{ParamSpec, ParamValues};

    #[derive(Debug, Default)]
    struct NoopEffect;

    static NOOP_META: EffectMetadata = EffectMetadata {
        id: "test.noop",
        display_name: "Noop",
        description: "passes input through",
        category: Category::Qa,
        version: 1,
    };

    impl Effect for NoopEffect {
        fn metadata(&self) -> &EffectMetadata { &NOOP_META }
        fn parameters(&self) -> &[ParamSpec] { &[] }
        fn capabilities(&self) -> Capabilities {
            Capabilities::cpu_only_deterministic()
        }
        fn apply(&self, _ctx: &mut Context, input: Frame, _p: &ParamValues) -> Result<Frame> {
            Ok(input)
        }
    }

    #[test]
    fn register_and_lookup() {
        let r = EffectRegistry::new();
        r.register_default::<NoopEffect>().unwrap();
        assert_eq!(r.len(), 1);
        assert!(r.get("test.noop").is_some());
        assert!(r.get("does.not.exist").is_none());
    }

    #[test]
    fn duplicate_register_errs() {
        let r = EffectRegistry::new();
        r.register_default::<NoopEffect>().unwrap();
        let e = r.register_default::<NoopEffect>();
        assert!(e.is_err());
    }
}
