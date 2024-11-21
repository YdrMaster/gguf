use super::{f16, _32};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ4NL {
    delta: f16,
    qs: [u16; _32 / 2],
}

impl_data_block! {
    IQ4NL = crate::types::IQ4NL;
    Self {
        delta: f16::ZERO,
        qs: [0; _32 / 2],
    }
}

impl Quantize<f32, _32> for IQ4NL {
    fn quantize(_data: &[f32; _32]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _32] {
        todo!()
    }
}
