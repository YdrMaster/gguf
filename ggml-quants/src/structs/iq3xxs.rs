use super::{f16, _256};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ3XXS {
    delta: f16,
    qs: [u16; 3 * _256 / 8],
}

impl_data_block! {
    IQ3XXS = crate::types::IQ3XXS;
    Self {
        delta: f16::ZERO,
        qs: [0; 3 * _256 / 8],
    }
}

impl Quantize<f32, _256> for IQ3XXS {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
