mod cast;
mod transpose;

use super::{compile_patterns, Content, DataPromise};
use ggus::GGmlType;
use regex::Regex;

pub(crate) enum Operator {
    FilterMetaKey(Regex),
    FilterTensorName(Regex),
    Cast(GGmlType),
    TransposeLinear(bool),
    ConcatLinear,
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
            TransposeLinear(mode) => self.transpose_linear(mode),
            ConcatLinear => self.concat_linear(),
        }
    }

    fn concat_linear(&mut self) {
        self.assert_llama();

        let _blk = self
            .meta_kvs
            .get("llama.block_count")
            .expect("missing block count")
            .value_reader()
            .read::<u64>()
            .unwrap_or_else(|e| panic!("failed to read block count: {e:?}"));

        todo!()
    }

    fn assert_llama(&self) {
        match self
            .meta_kvs
            .get("general.architecture")
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
