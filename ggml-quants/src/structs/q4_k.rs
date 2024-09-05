use super::{DeltaMin, _256};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct Q4K {
    delta_min: DeltaMin,
    scales: [u8; 12],
    qs: [u8; _256 / 2],
}

impl DataBlock for Q4K {
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        delta_min: DeltaMin::ZERO,
        scales: [0; 12],
        qs: [0; _256 / 2],
    };
}

impl Quantize<f32, _256> for Q4K {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
