mod cast;
mod distribute;
mod merge;
mod set_meta;
mod sort;

use super::{compile_patterns, Content, DataPromise};
use ggus::{GGmlType, GGufMetaDataValueType, GENERAL_ARCHITECTURE};
use regex::Regex;
use std::{collections::HashMap, sync::LazyLock};

pub(crate) enum Operator {
    FilterMetaKey(Regex),
    FilterTensorName(Regex),
    Cast(GGmlType),
    MergeLinear(bool),
    SetMeta(HashMap<String, (GGufMetaDataValueType, Vec<u8>)>),
    SortTensors,
    Distribute(usize),
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
            SetMeta(map) => self.set_meta(map),
            SortTensors => self.sort_tensors(),
            Distribute(n) => self.distribute(n),
        }
    }

    fn arch(&self) -> &str {
        self.meta_kvs
            .get(GENERAL_ARCHITECTURE)
            .map(|v| {
                assert_eq!(v.ty, GGufMetaDataValueType::String);
                v.value_reader().read_general_architecture_val()
            })
            .expect("missing architecture")
            .unwrap_or_else(|e| panic!("failed to read architecture: {e:?}"))
    }

    fn assert_llama(&self) {
        match self.arch() {
            "llama" => {}
            arch => todo!("unsupported architecture: {arch}"),
        }
    }
}

static BLK_TENSOR_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^blk\.(\d+)\.(\w+)\.weight$").unwrap());
