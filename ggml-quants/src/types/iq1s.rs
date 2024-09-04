use super::{f16, _256};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ1S {
    delta: f16,
    qs: [u8; _256 / 8],
    qh: [u16; _256 / 32],
}

impl DataBlock for IQ1S {
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        delta: f16::ZERO,
        qs: [0; _256 / 8],
        qh: [0; _256 / 32],
    };
}

impl Quantize<f32, _256> for IQ1S {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
