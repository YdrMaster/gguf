use super::{Content, Operator, LAYOUT_REFERENCE, LAYOUT_TRANSPOSED};
use crate::utils::{operator::LLAMA_TENSOR_DATA_LAYOUT, DataPromise, MetaValue};
use ggus::{DataFuture, GGufMetaDataValueType, GGufWriter};
use memmap2::MmapMut;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use regex::Regex;
use std::borrow::Cow;

impl Operator {
    pub fn transpose_linear(m: impl AsRef<str>) -> Self {
        let m = m.as_ref().trim();
        Self::TransposeLinear(match m.to_lowercase().as_str() {
            LAYOUT_TRANSPOSED | "yes" | "y" | "true" | "t" => true,
            LAYOUT_REFERENCE | "ref" | "default" | "no" | "n" | "false" | "f" => false,
            _ => panic!("unsupported transpose type: {m}"),
        })
    }
}

#[inline]
fn layout_meta_value<'a>(ty: bool) -> MetaValue<'a> {
    MetaValue {
        ty: GGufMetaDataValueType::String,
        value: {
            let mut buf = Vec::new();
            GGufWriter::new(&mut buf)
                .write_str(if ty {
                    LAYOUT_TRANSPOSED
                } else {
                    LAYOUT_REFERENCE
                })
                .unwrap();
            Cow::Owned(buf)
        },
    }
}

impl Content<'_> {
    pub(super) fn transpose_linear(&mut self, ty: bool) {
        self.assert_llama();

        use indexmap::map::Entry::*;
        match self.meta_kvs.entry(LLAMA_TENSOR_DATA_LAYOUT.into()) {
            Occupied(mut entry) => {
                match entry.get().value_reader().read_str().unwrap() {
                    LAYOUT_TRANSPOSED => {
                        if ty {
                            return;
                        }
                    }
                    LAYOUT_REFERENCE => {
                        if !ty {
                            return;
                        }
                    }
                    x => panic!("unsupported tensor data layout: {x}"),
                }
                *entry.get_mut() = layout_meta_value(ty);
            }
            Vacant(entry) => {
                if !ty {
                    return;
                }
                entry.insert(layout_meta_value(ty));
            }
        }

        let regex = Regex::new(r"^blk\.\d+\.(\w+)\.weight$").unwrap();
        for (name, tensor) in self.tensors.as_mut_slice() {
            let &[r, c] = tensor.shape.as_slice() else {
                continue;
            };
            match regex.captures(name) {
                Some(captures) => match &captures[1] {
                    "attn_output" | "ffn_down" => continue,
                    _ => {}
                },
                None => continue,
            }

            let data = tensor.data.clone();
            let count = tensor.ty.nbytes();

            tensor.shape = vec![c, r];
            let r = r as usize;
            let c = c as usize;

            tensor.data = DataPromise::lazy(move || {
                let src = data.get();
                let mut dst = MmapMut::map_anon(src.len()).unwrap();
                let ptr = dst.as_mut_ptr() as usize;
                (0..r * c).into_par_iter().for_each(|i| {
                    let src = src[(i % r * c) + i / r..].as_ptr();
                    let dst = (ptr + i) as *mut u8;
                    unsafe { std::ptr::copy_nonoverlapping(src, dst, count) };
                });
                dst
            });
        }
    }
}
