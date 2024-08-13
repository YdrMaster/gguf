use crate::name_pattern::compile_patterns;
use regex::Regex;

use super::Content;

pub enum Operator {
    FilterMetaKey(Regex),
    FilterTensorName(Regex),
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
        }
    }
}
