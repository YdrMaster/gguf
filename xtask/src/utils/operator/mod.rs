mod merge;
mod quantize;
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

pub(crate) enum Operator {
    ToLlama(Option<String>),
    FilterMetaKey(Regex),
    FilterTensorName(Regex),
    Cast { w: GGmlType, a: GGmlType },
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
            &Self::Cast { w, a } => {
                fn write_ty(f: &mut fmt::Formatter, ty: GGmlType) -> fmt::Result {
                    match ty {
                        GGmlType::F32 => write!(f, "32"),
                        GGmlType::F16 => write!(f, "16"),
                        GGmlType::I8 => write!(f, "8"),
                        GGmlType::I16 => write!(f, "i16"),
                        GGmlType::I32 => write!(f, "i32"),
                        GGmlType::I64 => write!(f, "i64"),
                        GGmlType::F64 => write!(f, "64"),
                        GGmlType::BF16 => write!(f, "bf16"),
                        _ => write!(f, "{ty:?}"),
                    }
                }
                write!(f, "cast: w")?;
                write_ty(f, w)?;
                write!(f, "a")?;
                write_ty(f, a)
            }
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
            Cast { w, a } => self.cast(w, a),
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
