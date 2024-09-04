use super::{f16, _256};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ2XXS {
    delta: f16,
    qs: [u16; _256 / 8],
}

impl DataBlock for IQ2XXS {
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        delta: f16::ZERO,
        qs: [0; _256 / 8],
    };
}

impl Quantize<f32, _256> for IQ2XXS {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
