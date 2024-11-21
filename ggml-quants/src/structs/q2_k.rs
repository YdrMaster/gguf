use super::{DeltaMin, _256};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct Q2K {
    scales: [u8; _256 / 16],
    qs: [u8; _256 / 4],
    delta_min: DeltaMin,
}

impl_data_block! {
    Q2K = crate::types::Q2K;
    Self {
        scales: [0; _256 / 16],
        qs: [0; _256 / 4],
        delta_min: DeltaMin::ZERO,
    }
}

impl Quantize<f32, _256> for Q2K {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
