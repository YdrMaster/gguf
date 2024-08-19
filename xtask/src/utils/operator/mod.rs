mod cast;
mod merge;

use super::{compile_patterns, Content, DataPromise};
use ggus::GGmlType;
use regex::Regex;

pub(crate) enum Operator {
    FilterMetaKey(Regex),
    FilterTensorName(Regex),
    Cast(GGmlType),
    MergeLinear(bool),
}

impl Operator {
    #[inline]
    pub fn filter_meta_key(p: impl AsRef<str>) -> Self {
        Self::FilterMetaKey(compile_patterns(p.as_ref()))
    }

    #[inline]
    pub fn filter_tensor_name(p: impl AsRef<str>) -> Self {
        Self::FilterTensorName(compile_patterns(p.as_ref()))
    }
}

impl Content<'_> {
    pub fn apply(&mut self, op: Operator) {
        use Operator::*;
        match op {
            FilterMetaKey(r) => self.meta_kvs.retain(|k, _| r.is_match(k)),
            FilterTensorName(r) => self.tensors.retain(|k, _| r.is_match(k)),
            Cast(ty) => self.cast(ty),
            MergeLinear(ty) => self.merge_linear(ty),
        }
    }

    fn assert_llama(&self) {
        match self
            .meta_kvs
            .get(GENERAL_ARCHITECTURE)
            .expect("missing architecture")
            .value_reader()
            .read_str()
        {
            Ok("llama") => {}
            Ok(arch) => todo!("unsupported architecture: {arch}"),
            Err(e) => panic!("failed to read architecture: {e:?}"),
        }
    }
}

const GENERAL_ARCHITECTURE: &str = "general.architecture";
const LLAMA_BLOCK_COUNT: &str = "llama.block_count";
