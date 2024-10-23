use super::{Content, Operator};
use crate::utils::DataPromise;
use ggus::{
    ggml_quants::{bf16, f16},
    DataFuture, GGmlType,
};
use memmap2::MmapMut;
use std::{alloc::Layout, ops::MulAssign};

impl Operator {
    #[inline]
    pub fn scale_token_embd(scale: &str) -> Self {
        Self::ScaleTokenEmbd(scale.parse().expect("invalid scale"))
    }
}

impl Content<'_> {
    pub(super) fn scale_token_embd(&mut self, scale: f64) {
        let Some(tensor) = self.tensors.get_mut("token_embd.weight") else {
            return;
        };
        let data = tensor.data.clone();
        tensor.data = match tensor.ty {
            GGmlType::F64 => DataPromise::lazy(move || scale_(data.get(), scale)),
            GGmlType::F32 => DataPromise::lazy(move || scale_(data.get(), scale as f32)),
            GGmlType::F16 => DataPromise::lazy(move || scale_(data.get(), f16::from_f64(scale))),
            GGmlType::BF16 => DataPromise::lazy(move || scale_(data.get(), bf16::from_f64(scale))),
            ty => todo!("unsupported tensor type: {ty:?}"),
        }
    }
}

fn scale_<T: MulAssign + Clone>(data: &[u8], scale: T) -> MmapMut {
    assert_eq!(data.len() % size_of::<T>(), 0);
    let len = data.len() / size_of::<T>();

    let mut ans = MmapMut::map_anon(Layout::array::<T>(len).unwrap().size()).unwrap();
    ans.copy_from_slice(data);

    let (&mut [], data, &mut []) = (unsafe { ans.align_to_mut::<T>() }) else {
        panic!("data not aligned")
    };
    for x in data {
        *x *= scale.clone();
    }

    ans
}
