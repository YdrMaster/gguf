use super::{Content, Operator};
use crate::utils::{DataPromise, MetaValue};
use ggus::{DataFuture, GGufMetaDataValueType, GGufWriter};
use memmap2::MmapMut;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    borrow::Cow,
    sync::{Arc, LazyLock},
};

impl Operator {
    pub fn transpose_linear(m: impl AsRef<str>) -> Self {
        let m = m.as_ref().trim();
        Self::TransposeLinear(match m.to_lowercase().as_str() {
            "transposed" | "yes" | "y" | "true" | "t" => true,
            "ref" | "reference" | "default" | "no" | "n" | "false" | "f" => false,
            _ => panic!("unsupported transpose type: {m}"),
        })
    }
}

#[inline]
const fn layout_str(ty: bool) -> &'static str {
    if ty {
        "transposed"
    } else {
        "reference"
    }
}

#[inline]
fn layout_meta_value<'a>(ty: bool) -> MetaValue<'a> {
    MetaValue {
        ty: GGufMetaDataValueType::String,
        value: {
            let mut buf = Vec::new();
            GGufWriter::new(&mut buf).write_str(layout_str(ty)).unwrap();
            Cow::Owned(buf)
        },
    }
}

impl Content<'_> {
    pub(super) fn transpose_linear(&mut self, ty: bool) {
        self.assert_llama();

        use indexmap::map::Entry::*;
        match self.meta_kvs.entry("llama.tensor_data_layout".into()) {
            Occupied(mut entry) => {
                if entry.get().value_reader().read_str().unwrap() == layout_str(ty) {
                    return;
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

        for (name, tensor) in self.tensors.as_mut_slice() {
            let &[r, c] = tensor.shape.as_slice() else {
                continue;
            };
            if !name.starts_with("blk.")
                || name.ends_with(".attn_output.weight")
                || name.ends_with(".ffn_down.weight")
            {
                continue;
            }

            let data = tensor.data.clone();
            let count = tensor.ty.nbytes();

            tensor.shape = vec![c, r];
            let r = r as usize;
            let c = c as usize;

            tensor.data = DataPromise::Lazy(Arc::new(LazyLock::new(move || {
                let src = data.get();
                let dst = MmapMut::map_anon(src.len()).unwrap();
                let ptr = dst.as_ptr() as usize;
                (0..r * c).into_par_iter().for_each(|i| {
                    let src = src[(i % r * c) + i / r..].as_ptr();
                    let dst = (ptr + i) as *mut u8;
                    unsafe { std::ptr::copy_nonoverlapping(src, dst, count) };
                });
                dst
            })));
        }
    }
}
