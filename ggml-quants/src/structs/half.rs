use super::{bf16, f16, _1};
use crate::{DataBlock, Quantize};

impl DataBlock for f16 {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout = digit_layout::types::F16;
    const COUNT: usize = _1;
    const ZEROS: Self = Self::ZERO;
}

impl Quantize<f32, _1> for f16 {
    #[inline]
    fn quantize(&[data]: &[f32; _1]) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const { assert!(Self::COUNT == _1) }
        f16::from_f32(data)
    }
    #[inline]
    fn dequantize(&self) -> [f32; _1] {
        [self.to_f32()]
    }
}

impl DataBlock for bf16 {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout = digit_layout::types::BF16;
    const COUNT: usize = _1;
    const ZEROS: Self = Self::ZERO;
}

impl Quantize<f32, _1> for bf16 {
    #[inline]
    fn quantize(&[data]: &[f32; _1]) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const { assert!(Self::COUNT == _1) }
        bf16::from_f32(data)
    }
    #[inline]
    fn dequantize(&self) -> [f32; _1] {
        [self.to_f32()]
    }
}
