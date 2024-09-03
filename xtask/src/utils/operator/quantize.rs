use super::{Content, DataPromise, Operator};
use ggus::{DataFuture, GGmlType as Ty};
use half::{bf16, f16};
use log::debug;
use memmap2::MmapMut;
use quantization::{QuantExt, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};
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
                "q4_0" => Ty::Q4_0,
                "q4_1" => Ty::Q4_1,
                "q5_0" => Ty::Q5_0,
                "q5_1" => Ty::Q5_1,
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

            debug!("Casting tensor {name} from {from:?} to {to:?}");
            tensor.ty = to;

            let data = tensor.data.clone();
            let row = tensor.shape[0];
            tensor.data = DataPromise::lazy(move || cast(row as _, data.get(), from, to));
        }
    }
}

#[rustfmt::skip]
fn cast(row: usize, data: &[u8], from: Ty, to: Ty) -> MmapMut {
    match from {
        Ty::F32 => match to {
            Ty::F32      => unreachable!(),
            Ty::F16      => quantize::<f16 , f32,  1>(data, row),
            Ty::Q4_0     => quantize::<Q4_0, f32, 32>(data, row),
            Ty::Q4_1     => quantize::<Q4_1, f32, 32>(data, row),
            Ty::Q5_0     => quantize::<Q5_0, f32, 32>(data, row),
            Ty::Q5_1     => quantize::<Q5_1, f32, 32>(data, row),
            Ty::Q8_0     => quantize::<Q8_0, f32, 32>(data, row),
            Ty::BF16     => quantize::<bf16, f32,  1>(data, row),
            _ => todo!(),
        },
        Ty::F16 => match to {
            Ty::F32      => dequantize::<f16 , f32,  1>(data),
            Ty::F16      => unreachable!(),
            Ty::Q4_0     =>   quantize::<Q4_0, f16, 32>(data, row),
            Ty::Q4_1     =>   quantize::<Q4_1, f16, 32>(data, row),
            Ty::Q5_0     =>   quantize::<Q5_0, f16, 32>(data, row),
            Ty::Q5_1     =>   quantize::<Q5_1, f16, 32>(data, row),
            Ty::Q8_0     =>   quantize::<Q8_0, f16, 32>(data, row),
            Ty::BF16     =>   quantize::<bf16, f16,  1>(data, row),
            _ => todo!(),
        },
        Ty::BF16 => match to {
            Ty::F32      => dequantize::<bf16, f32 ,  1>(data),
            Ty::F16      =>   quantize::<f16 , bf16,  1>(data, row),
            Ty::Q4_0     =>   quantize::<Q4_0, bf16, 32>(data, row),
            Ty::Q4_1     =>   quantize::<Q4_1, bf16, 32>(data, row),
            Ty::Q5_0     =>   quantize::<Q5_0, bf16, 32>(data, row),
            Ty::Q5_1     =>   quantize::<Q5_1, bf16, 32>(data, row),
            Ty::Q8_0     =>   quantize::<Q8_0, bf16, 32>(data, row),
            Ty::BF16     => unreachable!(),
            _ => todo!(),
        },
        _ => cast(row, &cast(row, data, from, Ty::F32), Ty::F32, to),
    }
}

fn quantize<Ext: QuantExt<T, N>, T, const N: usize>(data: &[u8], row: usize) -> MmapMut {
    let src = reslice::<T>(data);
    assert_eq!(src.len() % row, 0);
    assert_eq!(row % N, 0);
    let mut ans = malloc::<Ext>(src.len() / N);
    let dst = reslice_mut::<Ext>(&mut ans);
    Ext::quantize_slice(dst, src).unwrap();
    ans
}

fn dequantize<Ext: QuantExt<T, N>, T, const N: usize>(data: &[u8]) -> MmapMut {
    let src = reslice::<Ext>(data);
    let mut ans = malloc::<T>(src.len() * N);
    let dst = reslice_mut::<T>(&mut ans);
    Ext::dequantize_slice(dst, src).unwrap();
    ans
}

#[inline]
fn malloc<T>(len: usize) -> MmapMut {
    MmapMut::map_anon(Layout::array::<T>(len).unwrap().size()).unwrap()
}

#[inline]
fn reslice<T>(data: &[u8]) -> &[T] {
    let ([], data, []) = (unsafe { data.align_to() }) else {
        panic!("data is not aligned");
    };
    data
}

#[inline]
fn reslice_mut<T>(data: &mut [u8]) -> &mut [T] {
    let ([], data, []) = (unsafe { data.align_to_mut() }) else {
        panic!("data is not aligned");
    };
    data
}
