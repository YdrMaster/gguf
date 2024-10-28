mod cast;
mod merge;
mod set_meta;
mod sort;
mod to_llama;

use super::{compile_patterns, Content, DataPromise};
use ggus::{GGmlType, GGufMetaDataValueType, GGufMetaMapExt};
use regex::Regex;
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{self},
    sync::LazyLock,
};

#[allow(unused)]
pub(crate) enum Operator {
    ToLlama(Option<String>),
    FilterMetaKey(Regex),
    FilterTensorName(Regex),
    Cast {
        embd: Option<GGmlType>,
        norm: Option<GGmlType>,
        mat: Option<GGmlType>,
    },
    MergeLinear(bool),
    SetMeta(HashMap<String, (GGufMetaDataValueType, Vec<u8>)>),
    SortTensors,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::ToLlama(_extra) => write!(f, "to-llama"),
            Self::FilterMetaKey(regex) => write!(f, "filter-meta: {}", regex.as_str()),
            Self::FilterTensorName(regex) => write!(f, "filter-tensor: {}", regex.as_str()),
            &Self::Cast { .. } => write!(f, "cast"),
            &Self::MergeLinear(val) => {
                if val {
                    write!(f, "merge-linear",)
                } else {
                    write!(f, "split-linear")
                }
            }
            Self::SetMeta(map) => {
                write!(f, "set-meta: {} items", map.len())
            }
            Self::SortTensors => write!(f, "sort-tensors"),
        }
    }
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
            ToLlama(extra) => self.convert_to_llama(extra),
            FilterMetaKey(r) => self.meta_kvs.retain(|k, _| r.is_match(k)),
            FilterTensorName(r) => self.tensors.retain(|k, _| r.is_match(k)),
            Cast { embd, norm, mat } => self.cast(embd, norm, mat),
            MergeLinear(ty) => self.merge_linear(ty),
            SetMeta(map) => self.set_meta(map),
            SortTensors => self.sort_tensors(),
        }
    }

    fn assert_llama(&self) {
        match self.general_architecture().unwrap() {
            "llama" => {}
            arch => todo!("unsupported architecture: {arch}"),
        }
    }
}

static BLK_TENSOR_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^blk\.(\d+)\.(\w+)\.weight$").unwrap());

#[inline]
fn blk_tensor_name(i: impl fmt::Display, name: impl fmt::Display) -> Cow<'static, str> {
    format!("blk.{i}.{name}.weight").into()
}
