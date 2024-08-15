use super::{Content, DataPromise, Operator};
use ggus::{DataFuture, GGmlType};
use half::f16;
use memmap2::MmapMut;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{
    alloc::Layout,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::{Arc, LazyLock},
};

impl Operator {
    pub fn cast(t: impl AsRef<str>) -> Self {
        let t = t.as_ref().trim();
        Self::Cast(match t.to_lowercase().as_str() {
            "f16" | "fp16" | "half" => GGmlType::F16,
            "f32" | "fp32" | "float" => GGmlType::F32,
            _ => panic!("unsupported cast type: {t}"),
        })
    }
}

impl Content<'_> {
    pub(super) fn cast(&mut self, ty: GGmlType) {
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
}
