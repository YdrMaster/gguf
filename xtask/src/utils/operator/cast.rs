use super::{Content, DataPromise, Operator};
use ggus::{
    ggml_quants::{bf16, f16, QuantExt, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1},
    DataFuture, GGmlType as Ty, GGufMetaMapExt,
};
use log::debug;
use memmap2::MmapMut;
use regex::Regex;
use std::{alloc::Layout, collections::HashMap, sync::LazyLock};

impl Operator {
    #[inline]
    pub fn cast(types: &str) -> Self {
        static REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\w+):(\w+)").unwrap());
        Self::Cast(
            REGEX
                .captures_iter(types)
                .map(|captures| {
                    let key = captures[1].to_string();
                    let val = parse(&captures[2]);
                    (key, val)
                })
                .collect(),
        )
    }
}

impl Content<'_> {
    pub(super) fn cast(&mut self, types: HashMap<String, Ty>) {
        match self.general_architecture().unwrap() {
            "llama" => {
                let [mat, embd, norm] =
                    ["mat", "embd", "norm"].map(|name| types.get(name).copied());
                self.cast_(mat, |name, shape| match name {
                    "token_embd.weight" | "output.weight" => embd,
                    _ if name.ends_with("_norm.weight") => norm,
                    _ if shape.len() > 1 => mat,
                    _ => None,
                })
            }
            "clip" => {
                let [weight, bias] = ["weight", "bias"].map(|name| types.get(name).copied());
                self.cast_(weight, |name, _| {
                    if name.ends_with(".weight") {
                        weight
                    } else if name.ends_with(".bias") {
                        bias
                    } else {
                        None
                    }
                })
            }
            arch => panic!("Unsupported architecture: {arch}"),
        }
    }

    fn cast_(&mut self, main: Option<Ty>, mut ty: impl FnMut(&str, &[u64]) -> Option<Ty>) {
        if let Some(main) = main {
            self.name.encoding = Some(format!("{main:?}").into());
        }
        for (name, tensor) in self.tensors.as_mut_slice() {
            let from = tensor.ty;
            let to = ty(name, &tensor.shape);

            if let Some(to) = to.filter(|to| from != *to) {
                debug!("Casting tensor {name} from {from:?} to {to:?}");
                tensor.ty = to;

                let data = tensor.data.clone();
                let row = tensor.shape[0];
                tensor.data = DataPromise::lazy(move || cast(row as _, data.get(), from, to))
            }
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
            Ty::Q8_1     => quantize::<Q8_1, f32, 32>(data, row),
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
            Ty::Q8_1     =>   quantize::<Q8_1, f16, 32>(data, row),
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
            Ty::Q8_1     =>   quantize::<Q8_1, bf16, 32>(data, row),
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

#[rustfmt::skip]
fn parse(s: &str) -> Ty {
    match s.to_ascii_uppercase().as_str() {
        "F32"      => Ty::F32,
        "F16"      => Ty::F16,
        "Q4_0"     => Ty::Q4_0,
        "Q4_1"     => Ty::Q4_1,
        "Q5_0"     => Ty::Q5_0,
        "Q5_1"     => Ty::Q5_1,
        "Q8_0"     => Ty::Q8_0,
        "Q8_1"     => Ty::Q8_1,
        "Q2K"      => Ty::Q2K,
        "Q3K"      => Ty::Q3K,
        "Q4K"      => Ty::Q4K,
        "Q5K"      => Ty::Q5K,
        "Q6K"      => Ty::Q6K,
        "Q8K"      => Ty::Q8K,
        "IQ2XXS"   => Ty::IQ2XXS,
        "IQ2XS"    => Ty::IQ2XS,
        "IQ3XXS"   => Ty::IQ3XXS,
        "IQ1S"     => Ty::IQ1S,
        "IQ4NL"    => Ty::IQ4NL,
        "IQ3S"     => Ty::IQ3S,
        "IQ2S"     => Ty::IQ2S,
        "IQ4XS"    => Ty::IQ4XS,
        "I8"       => Ty::I8,
        "I16"      => Ty::I16,
        "I32"      => Ty::I32,
        "I64"      => Ty::I64,
        "F64"      => Ty::F64,
        "IQ1M"     => Ty::IQ1M,
        "BF16"     => Ty::BF16,
        "Q4_0_4_4" => Ty::Q4_0_4_4,
        "Q4_0_4_8" => Ty::Q4_0_4_8,
        "Q4_0_8_8" => Ty::Q4_0_8_8,
        _          => todo!(),
    }
}

#[test]
fn test_parse() {
    let Operator::Cast(types) = Operator::cast("embd:f16 mat:q8_0, norm:f32") else {
        unreachable!()
    };
    assert_eq!(types.len(), 3);
    assert_eq!(types.get("embd"), Some(&Ty::F16));
    assert_eq!(types.get("mat"), Some(&Ty::Q8_0));
    assert_eq!(types.get("norm"), Some(&Ty::F32));
}
