use super::Content;
use crate::{convert::DataPromise, name_pattern::compile_patterns};
use ggus::{DataFuture, GGmlType};
use half::f16;
use memmap2::MmapMut;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use regex::Regex;
use std::{
    alloc::Layout,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::{Arc, LazyLock},
};

pub enum Operator {
    FilterMetaKey(Regex),
    FilterTensorName(Regex),
    Cast(GGmlType),
    TransposeLinear,
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
            TransposeLinear => self.transpose_linear(),
            ConcatLinear => self.concat_linear(),
        }
    }

    fn cast(&mut self, ty: GGmlType) {
        self.assert_llama();

        for (name, tensor) in self.tensors.as_mut_slice() {
            if !name.ends_with("_norm.weight") && tensor.ty != ty {
                let from = tensor.ty;
                let to = ty;
                let data = tensor.data.clone();

                tensor.ty = ty;
                tensor.data = DataPromise::Lazy(Arc::new(LazyLock::new(move || {
                    use GGmlType as Ty;
                    let data = data.get();
                    match (from, to) {
                        (Ty::F32, Ty::F16) => cast(data, |&x| f16::from_f32(x)),
                        (Ty::F16, Ty::F32) => cast(data, |&x| f16::to_f32(x)),
                        (_, _) => todo!("unsupported cast: {from:?} -> {to:?}"),
                    }
                })));
            }
        }

        fn cast<T: Sync, U: Send>(data: &[u8], f: fn(&T) -> U) -> MmapMut {
            let len = data.len() / size_of::<T>();

            let size = Layout::array::<U>(len).unwrap().size();
            let mut ans = MmapMut::map_anon(size).unwrap();

            let (dst, src) = unsafe {
                (
                    from_raw_parts_mut(ans.as_mut_ptr().cast::<U>(), len),
                    from_raw_parts(data.as_ptr().cast::<T>(), len),
                )
            };
            dst.into_par_iter().zip(src).for_each(|(y, x)| *y = f(x));

            ans
        }
    }

    fn transpose_linear(&mut self) {
        self.assert_llama();

        let layout = self
            .meta_kvs
            .get("llama.tensor_data_layout")
            .map(|v| {
                v.value_reader()
                    .read_str()
                    .unwrap_or_else(|e| panic!("failed to read tensor data layout: {e:?}"))
            })
            .unwrap_or("reference");

        if layout.split(';').any(|s| s == "transposed") {
            return;
        }

        todo!()
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
