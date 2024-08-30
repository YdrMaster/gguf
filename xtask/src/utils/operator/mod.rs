﻿mod distribute;
mod merge;
mod quantize;
mod set_meta;
mod sort;

use super::{compile_patterns, Content, DataPromise};
use ggus::{GGmlType, GGufMetaDataValueType, GENERAL_ARCHITECTURE};
use regex::Regex;
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{self},
    sync::LazyLock,
};

pub(crate) enum Operator {
    FilterMetaKey(Regex),
    FilterTensorName(Regex),
    Cast { w: GGmlType, a: GGmlType },
    MergeLinear(bool),
    SetMeta(HashMap<String, (GGufMetaDataValueType, Vec<u8>)>),
    SortTensors,
    Distribute(usize),
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
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
            &Self::Distribute(n) => write!(f, "distribute: {n}"),
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
            FilterMetaKey(r) => self.meta_kvs.retain(|k, _| r.is_match(k)),
            FilterTensorName(r) => self.tensors.retain(|k, _| r.is_match(k)),
            Cast { w, a } => self.cast(w, a),
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

#[inline]
fn blk_tensor_name(i: impl fmt::Display, name: impl fmt::Display) -> Cow<'static, str> {
    format!("blk.{i}.{name}.weight").into()
}
