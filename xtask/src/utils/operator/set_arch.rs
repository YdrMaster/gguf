use super::{Content, Operator};
use crate::utils::MetaValue;
use ggus::GGufMetaMapExt;

impl Operator {
    #[inline]
    pub fn set_arch(arch: &str) -> Self {
        Self::SetArch(arch.into())
    }
}

impl Content<'_> {
    pub(super) fn set_arch(&mut self, new: &str) {
        let old = self.general_architecture().unwrap();
        if old == new {
            return;
        }
        let old = format!("{old}.");
        for (k, v) in std::mem::take(&mut self.meta_kvs) {
            if k == "general.architecture" {
                self.meta_kvs.insert(k, MetaValue::string(new));
            } else {
                self.meta_kvs.insert(
                    match k.strip_prefix(&old) {
                        Some(body) => format!("{new}.{body}").into(),
                        None => k,
                    },
                    v,
                );
            }
        }
    }
}
