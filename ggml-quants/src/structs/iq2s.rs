use super::{f16, _256};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ2S {
    delta: f16,
    qs: [u8; _256 / 4],
    qh: [u8; _256 / 32],
    scales: [u8; _256 / 32],
}

impl DataBlock for IQ2S {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout = crate::types::IQ1M;
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        delta: f16::ZERO,
        qs: [0; _256 / 4],
        qh: [0; _256 / 32],
        scales: [0; _256 / 32],
    };
}

impl Quantize<f32, _256> for IQ2S {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
