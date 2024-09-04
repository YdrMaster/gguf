use super::{f16, _256};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ3S {
    delta: f16,
    qs: [u8; _256 / 4],
    qh: [u8; _256 / 32],
    signs: [u8; _256 / 8],
    scales: [u8; _256 / 64],
}

impl DataBlock for IQ3S {
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        delta: f16::ZERO,
        qs: [0; _256 / 4],
        qh: [0; _256 / 32],
        signs: [0; _256 / 8],
        scales: [0; _256 / 64],
    };
}

impl Quantize<f32, _256> for IQ3S {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
