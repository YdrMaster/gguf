use super::_256;
use crate::{DataBlock, Quantize};
use half::f16;

#[repr(C)]
pub struct Q6K {
    ql: [u8; _256 / 2],
    qh: [u8; _256 / 4],
    scales: [u8; _256 / 16],
    delta: f16,
}

impl_data_block! {
    Q6K = crate::types::Q6K;
    Self {
        ql: [0; _256 / 2],
        qh: [0; _256 / 4],
        scales: [0; _256 / 16],
        delta: f16::ZERO,
    }
}

impl Quantize<f32, _256> for Q6K {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
