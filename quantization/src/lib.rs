mod half;
mod q8_0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizeError {
    Indivisible,
    LengthMismatch,
}

pub trait QuantBlock: Sized {
    const BLOCK_SIZE: usize;

    fn arr_len(n: usize) -> Result<usize, QuantizeError> {
        if n % Self::BLOCK_SIZE == 0 {
            Ok(n / Self::BLOCK_SIZE)
        } else {
            Err(QuantizeError::Indivisible)
        }
    }
    fn quantize(dst: &mut [Self], src: &[f32]) -> Result<(), QuantizeError>;
    fn dequantize(dst: &mut [f32], src: &[Self]) -> Result<(), QuantizeError>;

    fn quantize_f16(dst: &mut [Self], src: &[f16]) -> Result<(), QuantizeError> {
        let mut buf = vec![0.; src.len()];
        f16::dequantize(&mut buf, src)?;
        Self::quantize(dst, &buf)
    }
    fn dequantize_f16(dst: &mut [f16], src: &[Self]) -> Result<(), QuantizeError> {
        let mut buf = vec![0.; src.len()];
        Self::dequantize(&mut buf, src)?;
        f16::quantize(dst, &buf)
    }

    fn quantize_bf16(dst: &mut [Self], src: &[bf16]) -> Result<(), QuantizeError> {
        let mut buf = vec![0.; src.len()];
        bf16::dequantize(&mut buf, src)?;
        Self::quantize(dst, &buf)
    }
    fn dequantize_bf16(dst: &mut [bf16], src: &[Self]) -> Result<(), QuantizeError> {
        let mut buf = vec![0.; src.len()];
        Self::dequantize(&mut buf, src)?;
        bf16::quantize(dst, &buf)
    }
}

pub use ::half::{bf16, f16};
pub use q8_0::Q8_0;
