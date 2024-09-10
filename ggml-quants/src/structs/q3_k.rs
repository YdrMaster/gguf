use super::{f16, _256};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct Q3K {
    hmask: [u8; _256 / 8],
    qs: [u8; _256 / 4],
    scales: [u8; 12],
    delta: f16,
}

impl DataBlock for Q3K {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout = crate::types::IQ1M;
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        hmask: [0; _256 / 8],
        qs: [0; _256 / 4],
        scales: [0; 12],
        delta: f16::ZERO,
    };
}

impl Quantize<f32, _256> for Q3K {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
