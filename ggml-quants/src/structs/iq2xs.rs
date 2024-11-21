use super::{f16, _256};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ2XS {
    delta: f16,
    qs: [u16; _256 / 8],
    qh: [u8; _256 / 32],
}

impl_data_block! {
    IQ2XS = crate::types::IQ2XS;
    Self {
        delta: f16::ZERO,
        qs: [0; _256 / 8],
        qh: [0; _256 / 32],
    }
}

impl Quantize<f32, _256> for IQ2XS {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
