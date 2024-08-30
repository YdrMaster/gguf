use super::{Content, DataPromise, Operator};
use ggus::{DataFuture, GGmlType as Ty};
use half::{bf16, f16};
use log::info;
use memmap2::MmapMut;
use quantization::{QuantBlock, Q8_0};
use std::alloc::Layout;

impl Operator {
    pub fn quantize(t: impl AsRef<str>) -> Self {
        fn split(s: &str) -> Option<(&str, &str)> {
            s.strip_prefix('w')?.split_once('a')
        }
        fn parse(s: &str) -> Ty {
            match s {
                "16" | "f16" | "fp16" | "half" => Ty::F16,
                "32" | "f32" | "fp32" | "float" => Ty::F32,
                "bf16" => Ty::BF16,
                "q8_0" => Ty::Q8_0,
                _ => panic!("Unsupported type: {s}"),
            }
        }

        let t = t.as_ref().trim().to_lowercase();
        let (w, a) = split(&t).expect("Cast type must be in the format of `w_a_`");
        Self::Cast {
            w: parse(w),
            a: parse(a),
        }
    }
}

impl Content<'_> {
    pub(super) fn cast(&mut self, tw: Ty, ta: Ty) {
        self.assert_llama();

        self.name.encoding = Some(format!("{tw:?}").into());
        for (name, tensor) in self.tensors.as_mut_slice() {
            if tensor.shape.len() == 1 {
                continue;
            }
            let from = tensor.ty;
            let to = match &**name {
                "token_embd.weight" | "output.weight" => ta,
                _ if !name.ends_with("_norm.weight") => tw,
                _ => continue,
            };
            if from == to {
                continue;
            }

            info!("Casting tensor {name} from {from:?} to {to:?}");
            tensor.ty = to;

            let data = tensor.data.clone();
            tensor.data = DataPromise::lazy(move || cast(data.get(), from, to));
        }
    }
}

#[rustfmt::skip]
fn cast(data: &[u8], from: Ty, to: Ty) -> MmapMut {
    match (from, to) {
        (Ty::F32 , Ty::F16 ) => from_f32 ::<f16 >(data),
        (Ty::F32 , Ty::BF16) => from_f32 ::<bf16>(data),
        (Ty::F32 , Ty::Q8_0) => from_f32 ::<Q8_0>(data),

        (Ty::F16 , Ty::F32 ) =>   to_f32 ::<f16 >(data),
        (Ty::F16 , Ty::BF16) => from_f16 ::<bf16>(data),
        (Ty::F16 , Ty::Q8_0) => from_f16 ::<Q8_0>(data),

        (Ty::BF16, Ty::F32 ) =>   to_f32 ::<bf16>(data),
        (Ty::BF16, Ty::F16 ) => from_bf16::<f16 >(data),
        (Ty::BF16, Ty::Q8_0) => from_bf16::<Q8_0>(data),

        (Ty::Q8_0, Ty::F32 ) =>   to_f32 ::<Q8_0>(data),
        (Ty::Q8_0, Ty::F16 ) =>   to_f16 ::<Q8_0>(data),
        (Ty::Q8_0, Ty::BF16) =>   to_bf16::<Q8_0>(data),

        (_, _) if from == to => unreachable!(),
        (_, _) => todo!(),
    }
}

fn from_f32<T: QuantBlock>(data: &[u8]) -> MmapMut {
    let src = reslice::<f32>(data);
    let mut ans = malloc_quant::<T>(src.len());
    let dst = reslice_mut::<T>(&mut ans);
    T::quantize(dst, src).unwrap();
    ans
}

fn from_f16<T: QuantBlock>(data: &[u8]) -> MmapMut {
    let src = reslice::<f16>(data);
    let mut ans = malloc_quant::<T>(src.len());
    let dst = reslice_mut::<T>(&mut ans);
    T::quantize_f16(dst, src).unwrap();
    ans
}

fn from_bf16<T: QuantBlock>(data: &[u8]) -> MmapMut {
    let src = reslice::<bf16>(data);
    let mut ans = malloc_quant::<T>(src.len());
    let dst = reslice_mut::<T>(&mut ans);
    T::quantize_bf16(dst, src).unwrap();
    ans
}

fn to_f32<T: QuantBlock>(data: &[u8]) -> MmapMut {
    let src = reslice::<T>(data);
    let mut ans = malloc_dequant::<T, f32>(src.len());
    let dst = reslice_mut::<f32>(&mut ans);
    T::dequantize(dst, src).unwrap();
    ans
}

fn to_f16<T: QuantBlock>(data: &[u8]) -> MmapMut {
    let src = reslice::<T>(data);
    let mut ans = malloc_dequant::<T, f16>(src.len());
    let dst = reslice_mut::<f16>(&mut ans);
    T::dequantize_f16(dst, src).unwrap();
    ans
}

fn to_bf16<T: QuantBlock>(data: &[u8]) -> MmapMut {
    let src = reslice::<T>(data);
    let mut ans = malloc_dequant::<T, bf16>(src.len());
    let dst = reslice_mut::<bf16>(&mut ans);
    T::dequantize_bf16(dst, src).unwrap();
    ans
}

fn malloc_quant<T: QuantBlock>(len: usize) -> MmapMut {
    let len = T::arr_len(len).expect("data is indivisible to this type");
    let len = Layout::array::<T>(len).unwrap().size();
    MmapMut::map_anon(len).unwrap()
}

fn malloc_dequant<T: QuantBlock, U>(len: usize) -> MmapMut {
    let len = Layout::array::<U>(len * T::BLOCK_SIZE).unwrap().size();
    MmapMut::map_anon(len).unwrap()
}

fn reslice<T>(data: &[u8]) -> &[T] {
    let ([], data, []) = (unsafe { data.align_to() }) else {
        panic!("data is not aligned");
    };
    data
}

fn reslice_mut<T>(data: &mut [u8]) -> &mut [T] {
    let ([], data, []) = (unsafe { data.align_to_mut() }) else {
        panic!("data is not aligned");
    };
    data
}
