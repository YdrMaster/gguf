use super::_256;
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ1M {
    qs: [u8; _256 / 8],
    qh: [u8; _256 / 16],
    scales: [u8; _256 / 32],
}

impl DataBlock for IQ1M {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout = crate::types::IQ1M;
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        qs: [0; _256 / 8],
        qh: [0; _256 / 16],
        scales: [0; _256 / 32],
    };
}

impl Quantize<f32, _256> for IQ1M {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
