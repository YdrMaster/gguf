use super::{bf16, f16, _1};
use crate::{DataBlock, Quantize};

use digit_layout::types as ty;
impl_data_block!( f16 = ty:: F16;  f16::ZERO);
impl_data_block!(bf16 = ty::BF16; bf16::ZERO);

impl Quantize<f32, _1> for f16 {
    #[inline]
    fn quantize(&[data]: &[f32; _1]) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _1)
        }
        f16::from_f32(data)
    }
    #[inline]
    fn dequantize(&self) -> [f32; _1] {
        [self.to_f32()]
    }
}

impl Quantize<f32, _1> for bf16 {
    #[inline]
    fn quantize(&[data]: &[f32; _1]) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _1)
        }
        bf16::from_f32(data)
    }
    #[inline]
    fn dequantize(&self) -> [f32; _1] {
        [self.to_f32()]
    }
}
