use super::{DeltaMin, _256};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct Q2K {
    scales: [u8; _256 / 16],
    qs: [u8; _256 / 4],
    delta_min: DeltaMin,
}

impl DataBlock for Q2K {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout = crate::types::Q2K;
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        scales: [0; _256 / 16],
        qs: [0; _256 / 4],
        delta_min: DeltaMin::ZERO,
    };
}

impl Quantize<f32, _256> for Q2K {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
